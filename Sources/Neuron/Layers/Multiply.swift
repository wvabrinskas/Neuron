import Foundation
import NumSwift

/// A layer that performs element-wise multiplication between its input and the output of a linked layer.
/// 
/// `Multiply` is an arithmetic layer that combines two tensors by multiplying them element-wise,
/// with bias disabled by default. It links to another layer identified by `linkTo`.
public final class Multiply: ArithmeticLayer {
  /// Initializes a `Multiply` layer with the specified configuration.
  ///
  /// - Parameter inputSize: The size of the input tensor. Defaults to an empty `TensorSize`.
  /// - Parameter initializer: The weight initializer type to use. Defaults to `.heNormal`.
  /// - Parameter linkId: A unique string identifier for this layer. Defaults to a new UUID string.
  /// - Parameter linkTo: The identifier of the layer whose output will be multiplied with this layer's input.
  public init(inputSize: TensorSize = TensorSize(array: []),
              initializer: InitializerType = .heNormal,
              linkId: String = UUID().uuidString,
              inverse: Bool = false,
              linkTo: String) {
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: false,
               encodingType: .multiply,
               inverse: inverse,
               linkId: linkId,
               linkTo: linkTo)
  }

  /// Decodes a Multiply layer from a serialized model.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: An error if required values cannot be decoded.
  required public init(from decoder: Decoder) throws {
    try super.init(from: decoder)
  }

  override public func function(input: Tensor, other: Tensor) -> Tensor {
    input * other
  }
}

