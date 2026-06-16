import Foundation
import NumSwift

//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/4/22.
//

import Foundation
import NumSwift

/// A layer that performs element-wise addition by linking the output of another layer to its input.
/// This layer combines two tensor streams via addition, with no bias term applied.
public final class Add: ArithmeticLayer {
  /// Initializes an `Add` layer with the specified input size, initializer, and link identifiers.
  /// - Parameter inputSize: The size of the input tensor. Defaults to an empty `TensorSize`.
  /// - Parameter initializer: The weight initializer type to use. Defaults to `.heNormal`.
  /// - Parameter linkId: A unique string identifier for this layer link. Defaults to a new UUID string.
  /// - Parameter linkTo: The identifier of the layer whose output will be added to this layer's input.
  public init(inputSize: TensorSize = TensorSize(array: []),
              initializer: InitializerType = .heNormal,
              linkId: String = UUID().uuidString,
              inverse: Bool = false,
              linkTo: String) {
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: false,
               encodingType: .add,
               inverse: inverse,
               linkId: linkId,
               linkTo: linkTo)
  }

  /// Decodes an Add layer from a serialized model.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: An error if required values cannot be decoded.
  required public init(from decoder: Decoder) throws {
    try super.init(from: decoder)
  }

  /// Adds two tensors element-wise.
  ///
  /// - Parameters:
  ///   - input: The primary input tensor.
  ///   - other: The tensor from the linked layer.
  /// - Returns: Element-wise sum of `input` and `other`.
  override public func function(input: Tensor, other: Tensor) -> Tensor {
    input + other
  }
}

