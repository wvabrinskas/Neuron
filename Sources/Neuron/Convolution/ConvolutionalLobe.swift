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
  private var currentInputGradients: [[[Float]]] = []
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
                          inputSize: inputSize,
                          optimizer: optimizer,
                          learningRate: learningRate)
      filters.append(filter)
    }
  }
  
  public func feed(inputs: [[[Float]]], training: Bool) -> [[[Float]]] {
    //store inputs to calculate gradients for backprop
    forwardInputs = inputs
    
    var results: [[[Float]]] = [[[Float]]](repeating: [[0]], count: filters.count)
        
    filters.concurrentForEach { element, index in
      let filter = element
      let convolved = filter.apply(to: inputs,
                                   inputSize: inputSize) + bias
      
      //activate
      var activated: [Float] = []
      
      for c in 0..<convolved.count {
        let input = convolved[c]
        let neuron = neurons[index][c]
        
        activated.append(neuron.applyActivation(sum: input))
      }
      
      let reshapedActivated = activated.reshape(columns: inputSize.columns)
      results[index] = reshapedActivated
    }

    return results
  }

  public func adjustWeights(batchSize: Int) {
    filters.concurrentForEach { element, index in
      element.adjustWeights(batchSize: batchSize)
    }
  }
  
  public func calculateGradients(with deltas: [[[Float]]]) -> [[[Float]]] {
    currentInputGradients.removeAll()
    
    let flippedTransposed = filters.map { $0.flip180() }.transposed() as [[[[Float]]]]
        
    var inputGradientsForFilter: [[Float]] = []

    for i in 0..<deltas.count {
      let delta = deltas[i]
      
      //get activation derivatives
      let neuronsForDeltas = neurons[i]
      let activationDerivs = neuronsForDeltas.map { $0.activationDerivative }
      let activatedDeltas = delta.flatMap { $0 } * activationDerivs
      let reshapedDeltas = activatedDeltas.reshape(columns: inputSize.columns)
      
      for f in 0..<flippedTransposed.count {
        let filter = flippedTransposed[f]
        let kernel = filter[i]
        
        let gradientsForKernelIndex = reshapedDeltas.conv2D(kernel,
                                                            filterSize: (filterSize.rows, filterSize.columns),
                                                            inputSize: (inputSize.rows, inputSize.columns))
        
        if let currentGradientsForFilter = inputGradientsForFilter[safe: f] {
          let updatedGradientsForFilter = currentGradientsForFilter + gradientsForKernelIndex
          inputGradientsForFilter[f] = updatedGradientsForFilter
          currentInputGradients[f] = updatedGradientsForFilter.reshape(columns: inputSize.columns)
        } else {
          inputGradientsForFilter.append(gradientsForKernelIndex)
          currentInputGradients.append(gradientsForKernelIndex.reshape(columns: inputSize.columns))
        }
      }
      
      calculateFilterGradients(deltas: reshapedDeltas, index: i)
    }
            
    //return CURRENT calculated gradient add to exising gradients after
    return currentInputGradients
  }

  private func calculateFilterGradients(deltas: [[Float]], index: Int) {
    let filterGradients: [[[Float]]] = filters[index].gradients
    var newGradients: [[[Float]]] = []
    
    for inputIndex in 0..<forwardInputs.count {
      let forward2dInputs = forwardInputs[inputIndex]
      
      let shape = forward2dInputs.shape
      let rows = shape[safe: 1] ?? 0
      let columns = shape[safe: 0] ?? 0
      
      var updateFilters: [[Float]] = []
      
      if filterGradients[safe: inputIndex] != nil {
        //append to previous gradients will be dividing by batch size later
        updateFilters = filterGradients[inputIndex]
      }
      
      for r in 0..<rows - filterSize.rows {
        for c in 0..<columns - filterSize.columns {
          let gradient = deltas[r][c]
          
          for fr in 0..<filterSize.rows {
            let dataRow = Array(forward2dInputs[r + fr][c..<c + filterSize.1])
            let gradientRow = dataRow * gradient
            
            if let current = updateFilters[safe: fr] {
              let updated = current + gradientRow
              updateFilters[fr] = updated
            } else {
              updateFilters.append(gradientRow)
            }
          }
        }
      }
      
      newGradients.append(updateFilters)
    }
    
    filters[index].setGradients(gradients: newGradients)
  }

  public func zeroGradients() {
    self.currentInputGradients = []
    self.inputGradients = []
    self.filters.forEach { $0.zeroGradients() }
    self.neurons.forEach { $0.forEach { $0.zeroGradients() } }
  }
  
  public func clear() {
    self.inputGradients = []
    self.filters.forEach { $0.clear() }
    self.neurons.forEach { $0.forEach { $0.clear() } }
  }

}