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
              linkTo: String) {
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: false,
               encodingType: .multiply,
               linkId: linkId,
               linkTo: linkTo)
  }

  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let linkTo = try container.decodeIfPresent(String.self, forKey: .linkTo) ?? ""
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    self.init(linkId: linkId, linkTo: linkTo)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.linkTo = linkTo
  }
  
  override public func function(input: Tensor, other: Tensor) -> Tensor {
    input * other
  }
}

