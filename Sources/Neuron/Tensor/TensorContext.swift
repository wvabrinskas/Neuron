//
//  File.swift
//  
//
//  Created by William Vabrinskas on 7/31/22.
//

import Foundation

/// A context object that stores backpropagation logic for a tensor operation.
///
/// `TensorContext` holds a closure used during the backward pass to compute
/// gradients with respect to inputs, weights, and biases.
public struct TensorContext: Codable {
/// A tuple containing the computed gradients for the input, weight, and bias tensors.
  public typealias TensorBackpropResult = (input: Tensor, weight: Tensor, bias: Tensor)
/// A closure type that computes backpropagation gradients.
///
/// - Parameter inputs: The forward-pass input tensor.
/// - Parameter gradient: The incoming gradient tensor from the next layer.
/// - Parameter wrt: An optional tensor indicating the parameter to differentiate with respect to.
/// - Returns: A `TensorBackpropResult` containing gradients for input, weight, and bias.
  public typealias TensorContextFunction = (_ inputs: Tensor, _ gradient: Tensor, _ wrt: Tensor) -> TensorBackpropResult
  var backpropagate: TensorContextFunction
  
  /// Creates a tensor context with an optional custom backpropagation closure.
  ///
  /// - Parameter backpropagate: Closure computing input/weight/bias gradients.
  ///   When omitted, gradients are passed through to inputs and parameter
  ///   gradients are zero tensors.
  public init(backpropagate: TensorContextFunction? = nil) {
    let defaultFunction = { (input: Tensor, gradient: Tensor, wrt: Tensor?) in
      return (Tensor(gradient.storage, size: gradient.size), Tensor(), Tensor())
    }
    
    self.backpropagate = backpropagate ?? defaultFunction
  }
  
  /// Encodes tensor context metadata.
  ///
  /// `TensorContext` currently stores executable closures, so there is no
  /// serializable payload.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public func encode(to encoder: Encoder) throws {}
  
  /// Decodes a tensor context.
  ///
  /// Since closures are not serialized, decoding restores a default context.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  public init(from decoder: Decoder) throws {
    self = TensorContext()
  }
}
