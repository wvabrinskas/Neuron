//
//  Lobe.swift
//  Nameley
//
//  Created by William Vabrinskas on 12/22/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

public typealias WeightConstraint = ClosedRange<Float>

internal struct LobeCompileModel {
  var inputNeuronCount: Int
  var layerType: LobeModel.LayerType
  var fullyConnected: Bool = true
  var weightConstraint: WeightConstraint? = nil
  var initializer: Initializers = .xavierNormal
  var optimizer: OptimizerFunction? = nil
}

/// Class that contains a group of Neurons
public class Lobe {
  
  /// Neurons in the Lobe object
  public var neurons: [Neuron] = []
  public var layer: LobeModel.LayerType = .output
  public var activation: Activation = .none
  public var isNormalized: Bool = false
  
  private var initializer: Initializers = .xavierNormal
  private var weightConstraints: WeightConstraint? = nil

  /// default initializer
  /// - Parameter neurons: Neruons to control
  public init(neurons: [Neuron],
              activation: Activation = .none) {
    self.neurons = neurons
    self.activation = activation
  }
  
  public init(model: LobeModel,
              learningRate: Float) {
    
    let nuc = Nucleus(learningRate: learningRate,
                      bias: model.bias)
    
    var neurons: [Neuron] = []
    for _ in 0..<model.nodes {
      let neuron = Neuron(nucleus: nuc,
                          activation: model.activation)
      neurons.append(neuron)
    }
    self.neurons = neurons
    self.activation = model.activation
  }
  
  @discardableResult
  internal func compile(model: LobeCompileModel) -> [[Float]] {
    
    self.layer = model.layerType
    
    guard model.fullyConnected else {
      var layerWeights: [[Float]] = []

      for n in 0..<neurons.count {
        let neuron = neurons[n]
        
        if let optim = model.optimizer {
          neuron.addOptimizer(optimizer: optim)
        }
        
        neuron.add(input: 0, weight: 0)
        neuron.layer = model.layerType
        layerWeights.append([0])
      }
      return layerWeights
    }
    
    self.initializer = model.initializer
    self.weightConstraints = model.weightConstraint
    
    var layerWeights: [[Float]] = []
    
    neurons.forEach { neuron in
      neuron.layer = model.layerType
      
      var inputs: [Float] = []
      var weights: [Float] = []
      
      for _ in 0..<model.inputNeuronCount {
        var weight = initializer.calculate(input: self.neurons.count, out: model.inputNeuronCount)
        
        if let constrain = weightConstraints {
          let minBound = constrain.lowerBound
          let maxBound = constrain.upperBound
          weight = min(maxBound, max(minBound, weight))
        }
        
        inputs.append(0)
        weights.append(weight)
      }
      
      layerWeights.append(weights)
      
      let biasWeight = initializer.calculate(input: self.neurons.count, out: model.inputNeuronCount)
      
      if let optim = model.optimizer {
        neuron.addOptimizer(optimizer: optim)
      }

      neuron.initialize(weights: weights, inputs: inputs)
      neuron.biasWeight = biasWeight
    }
    
    return layerWeights
  }
  
  /// Feeds inputs into this layer
  /// - Parameter inputs: inputs into the layer
  /// - Returns: the result of the activation functions of the neurons
  public func feed(inputs: [Float], training: Bool) -> [Float] {
    //each input neuron gets the value directly mapped 1 : 1
    //not fully connected so each neuron has a different input
    //can not use getActivated function for this
    if self.layer == .input {
      guard inputs.count == self.neurons.count else {
        print("error")
        return []
      }
      
      for i in 0..<inputs.count {
        let input = inputs[i]
        let neuron = neurons[i]
        neuron.replaceInputs(inputs: [input])
      }
      
      return inputs
    }
  
    return getActivated(replacingInputs: inputs)
  }
  
  /// Gets the activated results for the current inputs or the passed in inputs at this layer
  /// - Parameter inputs: Optional inputs to replace at this layer. If this is left out this layer will activate against the currently set inputs
  /// - Returns: The activated results at this layer
  public func getActivated(replacingInputs inputs: [Float] = []) -> [Float] {
    
    //calculate dot products for hidden layers
    guard let existingNeuronInputs = self.neurons.first?.inputValues else {
      return []
    }
    
    var inputsToUse = existingNeuronInputs
    
    if inputs.count > 0 {
      inputsToUse = inputs
      
      self.neurons.forEach { neuron in
        neuron.replaceInputs(inputs: inputs)
      }
    }
    
    var activatedResults: [Float] = []
        
    let rows = inputsToUse.count
    let columns = neurons.count
    var layerWeights = neurons.flatMap { $0.weights }
    
    layerWeights.transpose(columns: columns, rows: rows)
    
    let dotProducts = inputsToUse.multiDotProduct(B: layerWeights,
                                                  columns: Int32(columns),
                                                  rows: Int32(rows))
    
    for i in 0..<dotProducts.count {
      let product = dotProducts[i]
      let neuron = neurons[i]
      
      let result = neuron.applyActivation(sum: product)
      activatedResults.append(result)
    }

    return activatedResults
  }
  
  /// Sets or updates the current lobes deltas
  /// - Parameters:
  ///   - deltas: the deltas to apply
  ///   - update: boolean to indicate if you add the incoming deltas or just set to the incoming deltas
  @discardableResult
  public func calculateGradients(with deltas: [Float]) -> [[Float]] {
    guard deltas.count == neurons.count else {
      return []
    }
    
    var gradients: [[Float]] = []
    for i in 0..<neurons.count {
      let delta = deltas[i]
      let neuron = neurons[i]
      neuron.calculateGradients(delta: delta)
      gradients.append(neuron.gradients)
    }
    
    return gradients
  }
  
  /// Calculates deltas for each neuron for the next layer in the network. Updates the current layer deltas with the input previous layer deltas.
  /// - Parameter previousLayerDeltas: Incoming delta's from the previous layer
  /// - Returns: The next set of deltas to be passed to the next layer in the backpropagation.
  public func calculateDeltas(inputs: [Float], previousLayerCount: Int) -> [Float] {

    guard self.layer != .input else {
      return []
    }

    var deltas: [Float] = []
    
    for p in 0..<previousLayerCount {
      var delta: Float = 0
      
      for n in 0..<neurons.count {
        let neuron = neurons[n]
        let inputWeight = neuron.weights[p]
        let neuronDelta = inputs[n]
        
        let currentNeuronDelta = neuronDelta * inputWeight //error' * weight -> calculates the error' w.r.t to weight
        
        delta += currentNeuronDelta
      }
      
      deltas.append(delta)
    }

    return deltas
  }
  
  /// Adjusts all the weights in all the neurons in this Lobe
  public func adjustWeights(batchSize: Int) {
    //dispatch queue breaks everything per usual...
    neurons.forEach { neuron in
      neuron.adjustWeights(self.weightConstraints, batchSize: batchSize)
    }
  }
  
  /// Clear all the neurons in this Lobe
  public func clear() {
    neurons.forEach { (neuron) in
      neuron.clear()
    }
  }
  
  public func zeroGradients() {
    neurons.forEach { neuron in
      neuron.zeroGradients()
    }
  }
  
  public func gradients() -> [[Float]] {
    return neurons.map { $0.gradients }
  }
  
  /// Backpropagation deltas at this specific layer
  /// - Returns: The deltas as floats
  public func deltas() -> [Float] {
    let deltas = neurons.compactMap { $0.delta }
    return deltas
  }
  
  /// Updates the parameters for each Neuron such as learning rate, bias, etc.
  /// - Parameter nucleus: Object that contains the parameters to update
  public func updateNucleus(_ nucleus: Nucleus) {
    neurons.forEach { (neuron) in
      neuron.updateNucleus(nucleus: nucleus)
    }
  }
  
}
