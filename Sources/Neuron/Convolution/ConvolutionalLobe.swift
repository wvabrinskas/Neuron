//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/16/22.
//

import Foundation
import NumSwift

public class ConvolutionalLobe: Lobe {
  
  private var filters: [[[Float]]] = []
  private let filterSize: (Int, Int)
  private let inputSize: (rows: Int, columns: Int, depth: Int)
  private let learningRate: Float
  
  private var forwardInputs: [Float] = []
  private var inputGradients: [[Float]] = []
  private var filterGradients: [[[Float]]] = []
  private var initializer: Initializers = .xavierNormal
  private var optimizer: OptimizerFunction?
  private var filterCount: Int = 1

  public init(model: ConvolutionalLobeModel,
              learningRate: Float,
              optimizer: OptimizerFunction? = nil,
              initializer: Initializers = .xavierNormal) {
    
    self.filterSize = model.filterSize
    self.inputSize = model.inputSize
    self.learningRate = learningRate
    self.initializer = initializer
    self.optimizer = optimizer
    self.filterCount = model.filterCount
    
    super.init(model: model, learningRate: learningRate)
    
    initializeFilters()
  }
  
  private func initializeFilters() {
    let distribution = NormalDistribution(mean: 0, deviation: 0.01)
    
    for _ in 0..<filterCount {
      
      var filter: [[Float]] = []
      
      for _ in 0..<filterSize.0 {
        var filterRow: [Float] = []
        
        for _ in 0..<filterSize.1 {
          let weight = distribution.nextFloat()
          filterRow.append(weight)
        }
        
        filter.append(filterRow)
      }
      
      filters.append(filter)
    }
  }
  
  public override func adjustWeights(batchSize: Int) {
    //adjust filter weights
    for i in 0..<filterGradients.count {
      adjustFilterFor(index: i)
    }
  }
  
  private func adjustFilterFor(index: Int) {
    var filter = filters[index]
    let gradients = filterGradients[index]
    
    var newFilters: [[Float]] = []
    
    for f in 0..<filterGradients.count {
      let currentFilterRow = filter[f]
      
      let filterGradient = gradients[f]

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
    
    filterGradients[index] = filter
  }
  
  //TODO: optimize this!
  public override func calculateGradients(with deltas: [Float]) -> [[Float]] {
    let activationDerivs = neurons.map { $0.activationDerivative }
    let activatedDeltas = deltas * activationDerivs

    let reshapedDeltas = activatedDeltas.reshape(columns: inputSize.columns)

    //Full Conv
    //sum of filters
    var inputGrads: [Float] = []
    for i in 0..<filters.count {
      let filter = filters[i]
      let filter180 = filter.flip180()
      inputGrads = inputGrads + reshapedDeltas.conv2D(filter180)
    }
    
    inputGradients = inputGrads.reshape(columns: inputSize.columns)
    
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
    
   // self.filterGradients = updateFilters
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
    //sum of filters

    var convolved: [Float] = []
    let reshapedInputs = inputs.reshape(columns: inputSize.columns)
    
    for i in 0..<filters.count {
      let filter = filters[i]
      let newConvolved = reshapedInputs.conv2D(filter)
      convolved = convolved + newConvolved
    }
        
    var activated: [Float] = []
    
    for i in 0..<convolved.count {
      let input = convolved[i]
      let neuron = neurons[i]
      
      activated.append(neuron.applyActivation(sum: input))
    }
    
    return activated
  }

}
