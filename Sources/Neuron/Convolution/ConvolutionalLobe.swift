//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/19/22.
//

import Foundation
import NumSwift

public typealias TensorSize = (rows: Int, columns: Int, depth: Int)

public protocol ConvolutionalSupportedLobe {
  var neurons: [[Neuron]] { get }
  var layer: LayerType { get }
  var activation: Activation { get }
  var outputSize: TensorSize { get }
  
  func feed(inputs: [[[Float]]], training: Bool) -> [[[Float]]]
  func calculateGradients(with deltas: [[[Float]]]) -> [[[Float]]]
  func adjustWeights(batchSize: Int)
  func zeroGradients()
  func clear()
}

public class ConvolutionalLobe: ConvolutionalSupportedLobe {
  public var neurons: [[Neuron]] = []
  public var layer: LayerType = .output
  public var activation: Activation = .none
  public var outputSize: TensorSize {
    return (inputSize.rows, inputSize.columns, filterCount)
  }

  private var filters: [Filter] = []
  private let filterSize: TensorSize
  private let inputSize: TensorSize
  private let learningRate: Float
  private var forwardInputs: [[[Float]]] = []
  private var inputGradients: [[[Float]]] = []
  private var initializer: Initializer
  private var optimizer: OptimizerFunction?
  private var filterCount: Int = 1
  private let bias: Float

  public init(model: ConvolutionalLobeModel,
              learningRate: Float,
              optimizer: OptimizerFunction? = nil,
              initializer: Initializer) {
    
    precondition(model.filterSize.depth == model.inputSize.depth, "input depth must equal filter depth")

    self.filterSize = model.filterSize
    self.inputSize = model.inputSize
    self.learningRate = learningRate
    self.initializer = initializer
    self.optimizer = optimizer
    self.filterCount = model.filterCount
    self.bias = model.bias
    
    var filterNeurons: [[Neuron]] = []
    
    for _ in 0..<filterCount {
      var neuronsForInput: [Neuron] = []
      let nuc = Nucleus(learningRate: learningRate,
                        bias: 0)
      
      for _ in 0..<inputSize.rows * inputSize.columns {
        let neuron = Neuron(nucleus: nuc,
                            activation: model.activation)
        neuronsForInput.append(neuron)
      }
      
      filterNeurons.append(neuronsForInput)
    }

    self.neurons = filterNeurons
    self.activation = model.activation
    
    initializeFilters()
  }
  
  private func initializeFilters() {    
    for _ in 0..<filterCount {
      let filter = Filter(size: filterSize,
                          inputSize: inputSize,
                          optimizer: optimizer,
                          initializer: initializer,
                          learningRate: learningRate,
                          bias: bias)
      filters.append(filter)
    }
  }
  
  public func feed(inputs: [[[Float]]], training: Bool) -> [[[Float]]] {
    //store inputs to calculate gradients for backprop
    if training {
      forwardInputs = inputs
    }
    
    var results: [[[Float]]] = []//[[[Float]]](repeating: [[0]], count: filters.count)
    
    var i = 0
    filters.forEach { filter in
      let convolved = filter.apply(to: inputs,
                                   inputSize: inputSize)
      
      //activate
      var activated: [Float] = []
      
      for c in 0..<convolved.count {
        let input = convolved[c]
        let neuron = neurons[i][c]
        
        activated.append(neuron.applyActivation(sum: input))
      }
      
      let reshapedActivated = activated.reshape(columns: inputSize.columns)
      results.append(reshapedActivated)
      i += 1
    }
        
//    filters.concurrentForEach { element, index in
//      results[index] = reshapedActivated
//    }

    return results
  }

  public func adjustWeights(batchSize: Int) {
    filters.concurrentForEach { element, index in
      element.adjustWeights(batchSize: batchSize)
    }
  }
  
  public func calculateGradients(with deltas: [[[Float]]]) -> [[[Float]]] {
    let result = concurrentCalculateGradients(deltas: deltas)
    return result
  }
  
  private func concurrentCalculateGradients(deltas: [[[Float]]]) -> [[[Float]]] {
    let flippedTransposed = filters.map { $0.flip180() }.transposed() as [[[[Float]]]]

    //2D array because each item is the result of conv2d which returns a 1D array
    var inputGradientsForFilterTotal: [[[Float]]] = Array(repeatElement(NumSwift.zerosLike((rows: inputSize.depth,
                                                                                            columns: inputSize.columns * inputSize.rows)),
                                                                        count: deltas.count))
    
    deltas.concurrentForEach { element, index in
      let delta = element
      let i = index
      
      //2D array because each item is the result of conv2d which returns a 1D array
      var inputGradientsForFilter: [[Float]] = inputGradientsForFilterTotal[i]
      
      //get activation derivatives
      let neuronsForDeltas = neurons[i]
      let activationDerivs = neuronsForDeltas.map { $0.activationDerivative }
      let activatedDeltas = delta.flatMap { $0 } * activationDerivs
      let reshapedDeltas = activatedDeltas.reshape(columns: inputSize.columns)
      
      for f in 0..<flippedTransposed.count {
        let filter = flippedTransposed[f]
        let kernel = filter[i]
        
        var gradientsForKernelIndex = reshapedDeltas.conv2D(kernel,
                                                            filterSize: (filterSize.rows, filterSize.columns),
                                                            inputSize: (inputSize.rows, inputSize.columns))
        
        //normalize gradients
        gradientsForKernelIndex.normalize(1.0)
        
        let currentGradientsForFilter = inputGradientsForFilter[f]
        let updatedGradientsForFilter = currentGradientsForFilter + gradientsForKernelIndex
        
        inputGradientsForFilter[f] = updatedGradientsForFilter
      }
      
      inputGradientsForFilterTotal[i] = inputGradientsForFilter
      
      calculateFilterGradients(deltas: reshapedDeltas, index: i)
    }
    
    let shape = (inputSize.depth, inputSize.columns * inputSize.rows)
    let zeros = NumSwift.zerosLike(shape)
    
    let total: [[Float]] = inputGradientsForFilterTotal.reduce(zeros) { partialResult, next in
      var new: [[Float]] = []
      
      for i in 0..<partialResult.count {
        let current = partialResult[i]
        let nextItem = next[i]
        if current.count == 0 {
          new.append(nextItem)
        } else {
          new.append(current + nextItem)
        }
      }
      
      return new
    }
    
    let result = total.map { $0.reshape(columns: inputSize.columns) }
    return result
  }

  private func calculateFilterGradients(deltas: [[Float]], index: Int) {
    let filterGradients: [[[Float]]] = filters[index].gradients
    var newGradients: [[[Float]]] = []
        
    for i in 0..<forwardInputs.count {
      let currentFilterGradients = filterGradients[safe: i] ?? []
      //we zero pad because the input image is zero padded when convolution is applied in the forward
      let forward2dInputs = forwardInputs[i].zeroPad()
      
      var result = NumSwift.conv2dValid(signal: forward2dInputs, filter: deltas)
      
      //add previous gradients
      for c in 0..<result.count {
        var currentResult = result[c]
        
        //normalize gradients
        currentResult.normalize(1.0)
        
        if let cRow = currentFilterGradients[safe: c] {
          result[c] = currentResult + cRow
        }
      }

      newGradients.append(result)
    }
    
    filters[index].setDeltas(deltas: deltas)
    filters[index].setGradients(gradients: newGradients)
  }

  public func zeroGradients() {
    self.inputGradients.removeAll(keepingCapacity: true)
    self.filters.forEach { $0.zeroGradients() }
    self.neurons.forEach { $0.forEach { $0.zeroGradients() } }
  }
  
  public func clear() {
    self.inputGradients.removeAll(keepingCapacity: true)
    self.filters.forEach { $0.clear() }
    self.neurons.forEach { $0.forEach { $0.clear() } }
  }

}
