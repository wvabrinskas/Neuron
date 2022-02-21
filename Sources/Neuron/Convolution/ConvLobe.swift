//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/19/22.
//

import Foundation
import NumSwift

public class ConvLobe {
  public var neurons: [[Neuron]] = []
  public var layer: LayerType = .output
  public var activation: Activation = .none
  
  private var filters: [[[Float]]] = []
  private let filterSize: (Int, Int)
  private let inputSize: (rows: Int, columns: Int, depth: Int)
  private let learningRate: Float
  private var forwardInputs: [[[Float]]] = []
  private var inputGradients: [[[Float]]] = []
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
    
    var filterNeurons: [[Neuron]] = []
    
    for _ in 0..<filterCount {
      var neuronsForInput: [Neuron] = []
      let nuc = Nucleus(learningRate: learningRate,
                        bias: model.bias)
      
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
  
  public func adjustWeights(batchSize: Int) {
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

  public func calculateGradients(with deltas: [[[Float]]]) -> [[[Float]]] {
    filterGradients.removeAll()
    
    var allInputGradients: [[[Float]]] = []
    
    //number of deltas should match the number of filters
    for i in 0..<deltas.count {
      let d = deltas[i].flatMap { $0 }
      let neuronsForDeltas = neurons[i]
      
      let activationDerivs = neuronsForDeltas.map { $0.activationDerivative }
      let activatedDeltas = d * activationDerivs

      let reshapedDeltas = activatedDeltas.reshape(columns: inputSize.columns)

      //Full Conv
      //sum of filters
      var inputGrads: [Float] = []
      for i in 0..<filters.count {
        let filter = filters[i]
        let filter180 = filter.flip180()
        inputGrads = inputGrads + reshapedDeltas.conv2D(filter180)
      }
      
      let currentInputGradients = inputGrads.reshape(columns: inputSize.columns)
            
      allInputGradients.append(currentInputGradients)
      
      calculateFilterGradients(deltas: reshapedDeltas, filterIndex: i)
    }

    inputGradients = allInputGradients
    
    return inputGradients
  }
  
  private func calculateFilterGradients(deltas: [[Float]], filterIndex: Int) {
    
    let forward2dInputs = forwardInputs[filterIndex]
    
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
    
    filterGradients.append(updateFilters)
  }
  
//  public func calculateDeltasForPreviousLayer(incomingDeltas: [Float], previousLayerCount: Int) -> [Float] {
//    return inputGradients.flatMap { $0 }
//  }
//
  
//  public func zeroGradients() {
//    self.inputGradients.removeAll()
//    self.filterGradients.removeAll()
//    self.neurons.forEach { $0.zeroGradients() }
//  }
  
//  public func clear() {
//    self.inputGradients.removeAll()
//    self.filterGradients.removeAll()
//    self.neurons.forEach { $0.clear() }
//  }
//
  public func feed(inputs: [[[Float]]], training: Bool) -> [[[Float]]] {
    //store inputs to calculate gradients for backprop
    forwardInputs = inputs
    
    //we need to know the input shape from the previous layer
    //using input size
    //sum of filters
    var results: [[[Float]]] = []

    for f in 0..<filters.count {
      let filter = filters[f]
      
      var convolved: [Float] = []
      for i in 0..<inputs.count {
        let input = inputs[i]
        let inputNuerons: [Neuron] = neurons[i]
        let newConvolved = input.conv2D(filter)

        if convolved.count == 0 {
          convolved = newConvolved
        } else {
          convolved = convolved + newConvolved
        }
      }
      
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

}
