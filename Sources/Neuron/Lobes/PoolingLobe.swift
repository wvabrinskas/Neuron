//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/18/22.
//

import Foundation
import NumSwift

public class PoolingLobe: Lobe {
  private let poolType: PoolType
  private let inputSize: (rows: Int, columns: Int)
  private var forwardPooledMaxIndicies: [(r: Int, c: Int)] = []
  private var forwardInputs: [Float] = []
  private var poolingGradients: [[Float]] = []
  
  public enum PoolType {
    case average, max
  }

  public init(model: PoolingLobeModel,
              learningRate: Float) {
    
    self.poolType = model.poolingType
    self.inputSize = model.inputSize
    
    super.init(model: model, learningRate: learningRate)
  
  }
  
  public override func feed(inputs: [Float], training: Bool) -> [Float] {
    //store inputs to calculate gradients for backprop

    forwardInputs = inputs
    
    //we need to know the input shape from the previous layer
    //using input size
    let reshapedInputs = inputs.reshape(columns: inputSize.columns)
    let pooled = pool(input: reshapedInputs)
      
    return pooled
  }
  
  public override func calculateGradients(with deltas: [Float]) -> [[Float]] {
    var modifiableDeltas = deltas
    //backprop pooling layer portion
    var pooledGradients = [Float].init(repeating: 0,
                                       count: forwardInputs.count).reshape(columns: self.inputSize.columns)
        
    forwardPooledMaxIndicies.forEach { index in
      pooledGradients[index.r][index.c] = modifiableDeltas.removeFirst()
    }
    
    self.poolingGradients = pooledGradients
    //TODO: reshape to be [] per neuron
    
    return pooledGradients
  }
  
  //no calculations happen here since there is math it's all done int eh calculate gradients function
  public override func calculateDeltasForPreviousLayer(incomingDeltas: [Float], previousLayerCount: Int) -> [Float] {
    return poolingGradients.flatMap { $0 }
  }
  
  public override func clear() {
    self.poolingGradients.removeAll()
    self.neurons.forEach { $0.clear() }
  }
  
  public override func zeroGradients() {
    self.poolingGradients.removeAll()
    self.neurons.forEach { $0.zeroGradients() }
  }
  
  public override func adjustWeights(batchSize: Int) {
    //no op
  }
  
  internal func pool(input: [[Float]]) -> [Float] {
    forwardPooledMaxIndicies.removeAll()

    var rowResults: [Float] = []
    var results: [Float] = []
        
    let rows = inputSize.rows
    let columns = inputSize.columns
        
    for r in stride(from: 0, through: rows - 1, by: 2) {
      rowResults = []
      for c in stride(from: 0, through: columns - 1, by: 2) {
        let current = input[r][c]
        let right = input[r + 1][c]
        let bottom = input[r][c + 1]
        let diag = input[r + 1][c + 1]
        
        if poolType == .max {
          let max = max(max(max(current, right), bottom), diag)
          forwardPooledMaxIndicies.append((r: r, c: c))
          rowResults.append(max)
        } else {
          let average = (current + right + bottom + diag) / 4
          rowResults.append(average)
        }
      }
            
      results.append(contentsOf: rowResults)
    }
        
    return results
  }
}
