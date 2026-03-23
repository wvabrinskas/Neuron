//
//  Divide.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/28/26.
//


import Foundation
import NumSwift

/// A layer that performs element-wise division between two linked layers in an arithmetic graph.
/// This layer divides the current input by the output of the linked layer, or inverts the operation
/// when `inverse` is set to `true`.
public final class Divide: ArithmeticLayer {
  
  /// Creates a new `Divide` layer with the specified configuration.
  ///
  /// - Parameter inputSize: The size of the input tensor. Defaults to an empty `TensorSize`.
  /// - Parameter initializer: The weight initializer type to use. Defaults to `.heNormal`.
  /// - Parameter linkId: A unique string identifier for this layer. Defaults to a new UUID string.
  /// - Parameter inverse: When `true`, inverts the division operation (i.e., divides the linked layer's output by this layer's input). Defaults to `false`.
  /// - Parameter linkTo: The identifier of the layer whose output will be used as the divisor.
  public init(inputSize: TensorSize = TensorSize(array: []),
              initializer: InitializerType = .heNormal,
              linkId: String = UUID().uuidString,
              inverse: Bool = false,
              linkTo: String) {
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: false,
               encodingType: .divide,
               inverse: inverse,
               linkId: linkId,
               linkTo: linkTo)
  }

  required public init(from decoder: Decoder) throws {
    try super.init(from: decoder)
  }

  override public func function(input: Tensor, other: Tensor) -> Tensor {
    input / other
  }
}
