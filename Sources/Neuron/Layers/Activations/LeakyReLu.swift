//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/29/22.
//

import Foundation
import NumSwift

/// Leaky Rectified Linear Unit (Leaky ReLU) activation layer
/// Applies the element-wise Leaky ReLU function: f(x) = max(αx, x) where α is the leak parameter
/// Unlike standard ReLU, Leaky ReLU allows a small gradient for negative values
/// This helps prevent the "dying ReLU" problem where neurons can become inactive
public final class LeakyReLu: BaseActivationLayer {
  /// The leak parameter (α) that determines the slope for negative values
  private var limit: Tensor.Scalar
  
  /// Initializes a Leaky ReLU activation layer
  /// For negative inputs, the function returns limit * x instead of 0
  /// - Parameter limit: The leak parameter (α), typically a small positive value like 0.01
  public init(limit: Tensor.Scalar = 0.01) {
    self.limit = limit
    
    super.init(type: .leakyRelu(limit: limit),
               encodingType: .leakyRelu)
  }
  
  /// Coding keys for serialization
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type,
         limit
  }
  
  /// Initializes Leaky ReLU layer from decoder for deserialization
  /// - Parameter decoder: Decoder containing serialized layer data
  /// - Throws: Decoding errors if deserialization fails
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let limit = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .limit) ?? 0.01
    self.init(limit: limit)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  /// Encodes the Leaky ReLU layer for serialization
  /// - Parameter encoder: Encoder to serialize layer data
  /// - Throws: Encoding errors if serialization fails
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(type, forKey: .type)
    try container.encode(limit, forKey: .limit)
  }
  
  /// Called when input size is set, configures output size
  /// For Leaky ReLU, output size equals input size as it's an element-wise operation
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
  }
}
