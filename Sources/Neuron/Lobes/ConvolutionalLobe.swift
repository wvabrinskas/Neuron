//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/16/22.
//

import Foundation
import NumSwift

public class ConvolutionalLobe: Lobe {
  
  private var filter: [[Float]] = []
  private let filterSize: (Int, Int)
  private let inputSize: (rows: Int, columns: Int)
  private let learningRate: Float
  
  private var forwardInputs: [Float] = []
  private var inputGradients: [[Float]] = []
  private var filterGradients: [[Float]] = []
  private var initializer: Initializers = .xavierNormal
  private var optimizer: OptimizerFunction?

  public init(model: ConvolutionalLobeModel,
              learningRate: Float,
              optimizer: OptimizerFunction? = nil,
              initializer: Initializers = .xavierNormal) {
    
    self.filterSize = model.filterSize
    self.inputSize = model.inputSize
    self.learningRate = learningRate
    self.initializer = initializer
    self.optimizer = optimizer
    
    super.init(model: model, learningRate: learningRate)
    
    initializeFilter()
  }
  
  private func initializeFilter() {
    let distribution = NormalDistribution(mean: 0, deviation: 0.01)
    
    for _ in 0..<filterSize.0 {
      var filterRow: [Float] = []
      
      for _ in 0..<filterSize.1 {
        let weight = distribution.nextFloat()
        filterRow.append(weight)
      }
      
      filter.append(filterRow)
    }
  }
  
  public override func adjustWeights(batchSize: Int) {
    //adjust filter weights
    var newFilters: [[Float]] = []
    
    for f in 0..<filterGradients.count {
      let currentFilterRow = filter[f]
      
      let filterGradient = filterGradients[f]

      var newFilter: [Float] = []
      if let optimizer = optimizer {
        
        for i in 0..<filterGradient.count {
          let g = filterGradient[i]
          let w = currentFilterRow[i]
          let newWeight = optimizer.run(weight: w, gradient: g)
          newFilter.append(newWeight)
        }
        
      } else {
        let adjustFilterGradient = filterGradient * learningRate
        newFilter = currentFilterRow - adjustFilterGradient
      }
      
      newFilters.append(newFilter)
    }

    filter = newFilters
  }
  
  //TODO: optimize this!
  public override func calculateGradients(with deltas: [Float]) -> [[Float]] {
    let activationDerivs = neurons.map { $0.activationDerivative }
    let activatedDeltas = deltas * activationDerivs

    let reshapedDeltas = activatedDeltas.reshape(columns: inputSize.columns)

    //Full Conv
    let filter180 = filter.flip180()
    inputGradients = reshapedDeltas.conv2D(filter180).reshape(columns: inputSize.columns)
    
    calculateFilterGradients(deltas: reshapedDeltas)

    return inputGradients
  }
  
  private func calculateFilterGradients(deltas: [[Float]]) {
    
    let forward2dInputs = forwardInputs.reshape(columns: inputSize.columns)
    
    let shape = forward2dInputs.shape
    let rows = shape[safe: 0] ?? 0
    let columns = shape[safe: 1] ?? 0
    
    var updateFilters: [[Float]] = [[Float]].init(repeating: [Float].init(repeating: 0,
                                                                          count: filterSize.0),
                                                  count: filterSize.1)
    for r in 0..<rows - filterSize.0 {
      for c in 0..<columns - filterSize.1 {
        let gradient = deltas[r][c]
        
        for fr in 0..<filterSize.1 {
          let dataRow = Array(forward2dInputs[r + fr][c..<c + filterSize.1])
          let gradientRow = dataRow * gradient
          let updated = updateFilters[fr] + gradientRow
          updateFilters[fr] = updated
        }
      }
    }
    
    self.filterGradients = updateFilters
  }
  
  public override func calculateDeltasForPreviousLayer(incomingDeltas: [Float], previousLayerCount: Int) -> [Float] {
    return inputGradients.flatMap { $0 }
  }
  
  public override func zeroGradients() {
    self.inputGradients.removeAll()
    self.filterGradients.removeAll()
    self.neurons.forEach { $0.zeroGradients() }
  }
  
  public override func clear() {
    self.inputGradients.removeAll()
    self.filterGradients.removeAll()
    self.neurons.forEach { $0.clear() }
  }
  
  public override func feed(inputs: [Float], training: Bool) -> [Float] {
    //store inputs to calculate gradients for backprop
    forwardInputs = inputs
    
    //we need to know the input shape from the previous layer
    //using input size
    let reshapedInputs = inputs.reshape(columns: inputSize.columns)
    let convolved = reshapedInputs.conv2D(filter)
        
    var activated: [Float] = []
    
    for i in 0..<convolved.count {
      let input = convolved[i]
      let neuron = neurons[i]
      
      activated.append(neuron.applyActivation(sum: input))
    }
    
    return activated
  }

}
