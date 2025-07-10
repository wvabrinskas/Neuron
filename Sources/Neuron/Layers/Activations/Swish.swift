//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/2/22.
//

import Foundation
import NumSwift

/// Swish activation layer (also known as SiLU - Sigmoid Linear Unit)
/// Applies the swish function: f(x) = x * sigmoid(x) = x / (1 + e^(-x))
/// Self-gated activation that can outperform ReLU in some deep networks
/// Smooth and non-monotonic, allowing for better gradient flow
public final class Swish: BaseActivationLayer {

  /// Initializes a Swish activation layer
  /// Swish is a smooth activation that can improve training in deep networks
  /// - Parameter inputSize: Optional input tensor size. Required for first layer in network
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               type: .swish,
               encodingType: .swish)
  }
  
  /// Coding keys for serialization
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  /// Initializes Swish layer from decoder for deserialization
  /// - Parameter decoder: Decoder containing serialized layer data
  /// - Throws: Decoding errors if deserialization fails
  convenience required public init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  /// Encodes the Swish layer for serialization
  /// - Parameter encoder: Encoder to serialize layer data
  /// - Throws: Encoding errors if serialization fails
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  /// Called when input size is set, configures output size
  /// For Swish, output size equals input size as it's an element-wise operation
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
  }
  
}

