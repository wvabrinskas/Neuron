//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/7/22.
//

import Foundation
import NumSwift

/// Accumulates gradients returning the average gradients for each layer w.r.t the weights and an array of gradients w.r.t to each input
/// Used for batch training where gradients from multiple samples need to be accumulated and averaged
public class GradientAccumulator {
  /// Number of gradient accumulation iterations
  private var iterations: Int = 0
  /// Accumulated gradients with respect to bias parameters for each layer
  private var biasGradients: [[Tensor]] = []
  /// Accumulated gradients with respect to weight parameters for each layer
  private var weightGradients: [[Tensor]] = []
  /// Accumulated gradients with respect to each top level input
  private var inputGradients: [Tensor] = []
  /// Thread synchronization lock for safe concurrent access
  private let lock = NSLock()
  
  /// A flag that when enabled will average the gradients when calling `accumulate`. Default: `true`
  public var average: Bool = true
  
  /// Removes all elements from the gradient arrays and resets iteration count
  /// This clears all accumulated gradients and prepares for a new accumulation cycle
  public func clear() {
    biasGradients.removeAll(keepingCapacity: true)
    weightGradients.removeAll(keepingCapacity: true)
    inputGradients.removeAll(keepingCapacity: true)
    iterations = 0
  }
  
  /// Inserts the gradients from a Tensor.Gradient into the accumulator
  /// - Parameter gradient: The gradient object containing input, weight, and bias gradients
  public func insert(_ gradient: Tensor.Gradient) {
    let inputGradient = gradient.input[safe: 0, Tensor()]
    let weightGradients = gradient.weights
    let biasGradients = gradient.biases
    self.insert(input: inputGradient, weights: weightGradients, biases: biasGradients)
  }
  
  /// Inserts individual gradient components into the accumulator
  /// This method is thread-safe and can be called concurrently
  /// - Parameters:
  ///   - input: Gradient with respect to the input
  ///   - weights: Gradients with respect to each layer's weights
  ///   - biases: Gradients with respect to each layer's biases
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
  
  /// Performs the averaging calculation on the weight and bias gradients
  /// Input gradients are returned as-is without averaging
  /// - Parameter clearAtEnd: If true, clears all accumulated gradients after computation
  /// - Returns: A Tensor.Gradient containing averaged weight/bias gradients and all input gradients
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
    
    return .init(input: inputGradients, weights: weight, biases: bias)
  }
}