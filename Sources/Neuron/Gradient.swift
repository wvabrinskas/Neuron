//
//  File.swift
//
//
//  Created by William Vabrinskas on 6/7/22.
//

import Foundation
import NumSwift
import Atomics

/// Accumulates gradients returning the average gradients for each layer w.r.t the weights and an array of gradients w.r.t to each input
public class GradientAccumulator {
  private var iterations = ManagedAtomic<Int>(0)
  private var biasGradients: [Tensor] = []//gradients w.r.t to each layer's weights
  private var weightGradients: [Tensor] = []//gradients w.r.t to each layer's weights
  private var inputGradients: [Tensor] = [] //gradients w.r.t to each top level input

  private let lock = NSLock()
  
  /// A flag that when enabled will average the gradients when calling `accumulate`. Default: `true`
  public var average: Bool = true
  
  /// Removes all elements from the gradient arrays
  public func clear() {
    biasGradients.removeAll(keepingCapacity: true)
    weightGradients.removeAll(keepingCapacity: true)
    inputGradients.removeAll(keepingCapacity: true)
    iterations.store(0, ordering: .relaxed)
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
  ///   - biases: Gradients WRT to each layer's biases
  public func insert(input: Tensor, weights: [Tensor], biases: [Tensor]) {
    
    iterations.wrappingIncrement(by: 1, ordering: .relaxed)
    
    lock.with {
      
      if weightGradients.isEmpty {
        weightGradients = weights
      } else {
        weightGradients = weightGradients + weights
      }
      
      if biasGradients.isEmpty {
        biasGradients = biases
      } else {
        biasGradients = biasGradients + biases
      }
      inputGradients.append(input)
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
    
    var weight: [Tensor] = weightGradients
    var bias: [Tensor] = biasGradients
    
    let iterationsAtomic = iterations.load(ordering: .acquiring)
    if iterationsAtomic > 1 && average {
      weight = weightGradients / iterationsAtomic.asTensorScalar
      bias = biasGradients / iterationsAtomic.asTensorScalar
    }
        // average the gradients
    return .init(input: inputGradients, weights: weight, biases: bias)
  }
}
