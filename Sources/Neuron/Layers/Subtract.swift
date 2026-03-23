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

  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let linkTo = try container.decodeIfPresent(String.self, forKey: .linkTo) ?? ""
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    let inverse = try container.decodeIfPresent(Bool.self, forKey: .inverse) ?? false
    self.init(linkId: linkId, inverse: inverse, linkTo: linkTo)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.linkTo = linkTo
  }
  
  override public func function(input: Tensor, other: Tensor) -> Tensor {
    input - other
  }
}
