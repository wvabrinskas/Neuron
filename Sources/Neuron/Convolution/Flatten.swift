//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/22/22.
//

import Foundation
import NumSwift

public class Flatten {
  private var inputSize: (Int, Int, Int) = (0,0,0)
  
  public func feed(inputs: [[[Float]]]) -> [Float] {
    let shape = inputs.shape
    if let r = shape[safe: 1],
        let c = shape[safe: 0],
       let d = shape[safe: 2] {
      inputSize = (r, c, d)
    }
    
    return inputs.flatMap { $0.flatMap { $0 } }
  }
  
  public func backpropagate(deltas: [Float]) -> [[[Float]]] {
    let batchedDeltas = deltas.batched(into: inputSize.0 * inputSize.1)
    let gradients = batchedDeltas.map { $0.reshape(columns: inputSize.1) }
    return gradients
  }
}

