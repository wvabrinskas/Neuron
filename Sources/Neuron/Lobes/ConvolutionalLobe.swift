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
  
  private var forwardInputs: [Float] = []
  private var inputGradients: [[Float]] = []
  private var filterGradients: [[Float]] = []

  public init(model: ConvolutionalLobeModel,
              learningRate: Float) {
    
    self.filterSize = model.filterSize
    self.inputSize = model.inputSize
    
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
  
  //TODO: optimize this!
  public override func calculateGradients(with deltas: [Float]) -> [[Float]] {
    
    //TODO: calc derivative of activation as well
    
    let filter180 = filter.flip180()
    
    let reshapedDeltas = deltas.reshape(columns: inputSize.columns)
    
    inputGradients = reshapedDeltas.conv2D(filter180).reshape(columns: 1)
    
    var forward2dInputs = forwardInputs.reshape(columns: inputSize.columns)
    
    let deltas2d: [[Float]] = [[1,1,1],
                               [1,1,1],
                               [1,1,1]] //deltas.reshape (columns: (inputSize.columns / 2) + 1)
    
    let filterGradients = forward2dInputs.conv2D(deltas2d)
    
    return inputGradients
  }
  
  public override func calculateDeltasForPreviousLayer(incomingDeltas: [Float], previousLayerCount: Int) -> [Float] {
    return inputGradients.flatMap { $0 }
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
