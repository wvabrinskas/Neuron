//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/7/22.
//

import Foundation
import NumSwift

/// Accumulates gradients returning the average gradients for each layer w.r.t the weights and an array of gradients w.r.t to each input
public class GradientAccumulator {
  private var iterations: Int = 0
  private var biasGradients: [[Tensor]] = []//gradients w.r.t to each layer's weights
  private var weightGradients: [[Tensor]] = []//gradients w.r.t to each layer's weights
  private var inputGradients: [Tensor] = [] //gradients w.r.t to each top level input
  private let lock = NSLock()
  
  /// A flag that when enabled will average the gradients when calling `accumulate`. Default: `true`
  public var average: Bool = true
  
  /// Removes all elements from the gradient arrays
  public func clear() {
    biasGradients.removeAll(keepingCapacity: true)
    weightGradients.removeAll(keepingCapacity: true)
    inputGradients.removeAll(keepingCapacity: true)
    iterations = 0
  }
  
  /// Inserts the gradients into the accumulator
  /// - Parameter gradient: The gradient to add to the accumulator
  public func insert(_ gradient: Tensor.Gradient) {
    let inputGradient = gradient.input[safe: 0, Tensor()]
    let weightGradients = gradient.weights
    let biasGradients = gradient.biases
    self.insert(input: inputGradient, weights: weightGradients, biases: biasGradients)
  }
  
  /// Inserts the gradients into the accumulator
  /// - Parameters:
  ///   - input: Gradient WRT to the input
  ///   - weights: Gradients WRT to each layer's weights
  ///   - index: index to insert the gradient.
  public func insert(input: Tensor, weights: [Tensor], biases: [Tensor]) {
    let newWeightGradients = weights
    let newInputGradient = input
    let newBiasGradient = biases
    
    lock.with {
      iterations += 1
      
      weightGradients.append(newWeightGradients)
      biasGradients.append(newBiasGradient)
      inputGradients.append(newInputGradient)
    }
  }
  
  /// Performs the averaging calculation on the weight gradients.
  /// Does not perform an averaging calculation on the input gradients
  /// - Parameter clearAtEnd: will erase all the accumulated gradients thus far.
  /// - Returns: The average gradients w.r.t to each layers weights and the gradient w.r.t the each input given.
  public func accumulate(clearAtEnd: Bool = false) -> Tensor.Gradient {
    defer {
      if clearAtEnd {
        clear()
      }
    }
    
    let firstW = weightGradients.removeFirst()
    let weightSum = weightGradients.reduce(firstW, +)
    
    let firstBias = biasGradients.removeFirst()
    let biasSum = biasGradients.reduce(firstBias, +)
    
    var weight = weightSum
    var bias = biasSum
    
    if iterations > 1 && average {
      weight = weight / iterations.asTensorScalar
      bias = bias / iterations.asTensorScalar
    }
        // average the gradients
    return .init(input: inputGradients, weights: weight, biases: bias)
  }
}
