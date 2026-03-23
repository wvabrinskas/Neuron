//
//  Subtract.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/28/26.
//


import Foundation
import NumSwift

/// A neural network layer that performs element-wise subtraction between two linked layers.
/// This layer subtracts the output of a linked layer from its input (or vice versa if `inverse` is true).
public final class Subtract: ArithmeticLayer {
  /// Initializes a Subtract layer with the specified configuration.
  /// - Parameter inputSize: The size of the input tensor. Defaults to an empty `TensorSize`.
  /// - Parameter initializer: The weight initializer type to use. Defaults to `.heNormal`.
  /// - Parameter linkId: A unique string identifier for this layer. Defaults to a new UUID string.
  /// - Parameter inverse: When `true`, reverses the order of subtraction. Defaults to `false`.
  /// - Parameter linkTo: The identifier of the layer whose output will be subtracted.
  public init(inputSize: TensorSize = TensorSize(array: []),
              initializer: InitializerType = .heNormal,
              linkId: String = UUID().uuidString,
              inverse: Bool = false,
              linkTo: String) {
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: false,
               encodingType: .subtract,
               inverse: inverse,
               linkId: linkId,
               linkTo: linkTo)
  }

  required public init(from decoder: Decoder) throws {
    try super.init(from: decoder)
  }

  override public func function(input: Tensor, other: Tensor) -> Tensor {
    input - other
  }
}
