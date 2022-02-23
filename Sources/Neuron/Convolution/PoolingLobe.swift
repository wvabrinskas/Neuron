//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/18/22.
//

import Foundation
import NumSwift

public class PoolingLobe: ConvolutionalSupportedLobe {
  public var neurons: [[Neuron]] = [] //pooling lobes dont need neurons
  public var layer: LayerType = .hidden
  public let activation: Activation = .none
  
  private let poolType: PoolType
  private var inputSize: TensorSize = (0,0,0)
  private var forwardPooledMaxIndicies: [[(r: Int, c: Int)]] = []
  private var forwardInputs: [[[Float]]] = []
  private var poolingGradients: [[[Float]]] = []
  
  public enum PoolType {
    case average, max
  }

  public init(model: PoolingLobeModel) {
    self.poolType = model.poolingType
  }

  //no calculations happen here since there is math it's all done int eh calculate gradients function
  public func calculateDeltasForPreviousLayer() -> [[[Float]]] {
    return poolingGradients
  }
  
  public func feed(inputs: [[[Float]]], training: Bool) -> [[[Float]]] {
    forwardPooledMaxIndicies.removeAll()

    let inputShape = inputs.shape
    
    if let r = inputShape[safe: 1],
       let c = inputShape[safe: 0],
       let d = inputShape[safe: 2] {
      inputSize = (r, c, d)
    }
    
    forwardInputs = inputs
    let results = inputs.map { pool(input: $0) }
    return results
  }
  
  public func calculateGradients(with deltas: [[[Float]]]) -> [[[Float]]] {
    poolingGradients.removeAll()
    
    for i in 0..<deltas.count {
      let delta = deltas[i].flatMap { $0 }
      var modifiableDeltas = delta
      
      var pooledGradients = [Float].init(repeating: 0,
                                         count: inputSize.rows * inputSize.columns).reshape(columns: inputSize.columns)
          
      let indicies = forwardPooledMaxIndicies[i]
      
      indicies.forEach { index in
        pooledGradients[index.r][index.c] = modifiableDeltas.removeFirst()
      }
      
      poolingGradients.append(pooledGradients)
    }
    
    return poolingGradients
  }
  
  public func clear() {
    self.poolingGradients.removeAll()
    self.neurons.forEach { $0.forEach { $0.clear() } }
  }
  
  public func zeroGradients() {
    self.poolingGradients.removeAll()
    self.neurons.forEach { $0.forEach { $0.zeroGradients() } }
  }
  
  public func adjustWeights(batchSize: Int) {
    //no op on pooling layer
  }
  
  internal func pool(input: [[Float]]) -> [[Float]] {
    var rowResults: [Float] = []
    var results: [[Float]] = []
    var pooledIndicies: [(r: Int, c: Int)] = []
        
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
          pooledIndicies.append((r: r, c: c))
          rowResults.append(max)
        } else {
          let average = (current + right + bottom + diag) / 4
          rowResults.append(average)
        }
      }
      
      results.append(rowResults)
    }
    
    forwardPooledMaxIndicies.append(pooledIndicies)
        
    return results
  }
}
