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
  func calculateDeltasForPreviousLayer() -> [[[Float]]]
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
  private var initializer: Initializers = .xavierNormal
  private var optimizer: OptimizerFunction?
  private var filterCount: Int = 1
  private let bias: Float

  public init(model: ConvolutionalLobeModel,
              learningRate: Float,
              optimizer: OptimizerFunction? = nil,
              initializer: Initializers = .xavierNormal) {
    
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
                          optimizer: optimizer,
                          learningRate: learningRate)
      filters.append(filter)
    }
  }
  
  public func feed(inputs: [[[Float]]], training: Bool) -> [[[Float]]] {
    //store inputs to calculate gradients for backprop
    forwardInputs = inputs
    
    //we need to know the input shape from the previous layer
    //using input size
    //sum of filters
    var results: [[[Float]]] = []

    for f in 0..<filters.count {
      let filter = filters[f]
      let convolved = filter.apply(to: inputs) + bias
      
      //activate
      var activated: [Float] = []
      
      for c in 0..<convolved.count {
        let input = convolved[c]
        let neuron = neurons[f][c]
        
        activated.append(neuron.applyActivation(sum: input))
      }
      
      let reshapedActivated = activated.reshape(columns: inputSize.columns)
      results.append(reshapedActivated)
    }

    return results
  }

  public func adjustWeights(batchSize: Int) {
    filters.forEach { $0.adjustWeights() }
  }
  
  //TODO: optimize this!
  public func calculateGradients(with deltas: [[[Float]]]) -> [[[Float]]] {
    
    let flippedTransposed = filters.map { $0.flip180() }.transposed() as [[[[Float]]]]
        
    var inputGradientsForFilter: [[Float]] = []//indexed off of current filter index

    for i in 0..<deltas.count {
      let delta = deltas[i].flatMap { $0 }
      
      //get activation derivatives
      let neuronsForDeltas = neurons[i]
      let activationDerivs = neuronsForDeltas.map { $0.activationDerivative }
      let activatedDeltas = delta * activationDerivs
      let reshapedDeltas = activatedDeltas.reshape(columns: inputSize.columns)
      
      for f in 0..<flippedTransposed.count {
        let filter = flippedTransposed[f]
        let kernel = filter[i]
        
        let gradientsForKernelIndex = reshapedDeltas.conv2D(kernel)
        
        if let currentGradientsForFilter = inputGradientsForFilter[safe: f] {
          let updatedGradientsForFilter = currentGradientsForFilter + gradientsForKernelIndex
          inputGradientsForFilter[f] = updatedGradientsForFilter
        } else {
          inputGradientsForFilter.append(gradientsForKernelIndex)
        }
      }
      
      calculateFilterGradients(deltas: reshapedDeltas, index: i)
    }
    
    inputGradients = inputGradientsForFilter.map { $0.reshape(columns: inputSize.columns) }
    
    return inputGradients
  }
  //TODO: figure out how to calculate gradients for new Filter
  private func calculateFilterGradients(deltas: [[Float]], index: Int) {
    var filterGradients: [[[Float]]] = []
    
    for inputIndex in 0..<forwardInputs.count {
      let forward2dInputs = forwardInputs[inputIndex]
      
      let shape = forward2dInputs.shape
      let rows = shape[safe: 1] ?? 0
      let columns = shape[safe: 0] ?? 0
      
      var updateFilters: [[Float]] = [[Float]].init(repeating: [Float].init(repeating: 0,
                                                                            count: filterSize.rows),
                                                    count: filterSize.columns)
      for r in 0..<rows - filterSize.rows {
        for c in 0..<columns - filterSize.columns {
          let gradient = deltas[r][c]
          
          for fr in 0..<filterSize.rows {
            let dataRow = Array(forward2dInputs[r + fr][c..<c + filterSize.1])
            let gradientRow = dataRow * gradient
            let updated = updateFilters[fr] + gradientRow
            updateFilters[fr] = updated
          }
        }
      }
      
      filterGradients.append(updateFilters)
    }
    
    filters[index].setGradients(gradients: filterGradients)
  }
  
  public func calculateDeltasForPreviousLayer() -> [[[Float]]] {
    return inputGradients
  }

  public func zeroGradients() {
    self.inputGradients.removeAll()
    self.filters.forEach { $0.zeroGradients() }
    self.neurons.forEach { $0.forEach { $0.zeroGradients() } }
  }
  
  public func clear() {
    self.inputGradients.removeAll()
    self.filters.forEach { $0.clear() }
    self.neurons.forEach { $0.forEach { $0.clear() } }
  }

}
