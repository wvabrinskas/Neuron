//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/2/22.
//

import Foundation
import NumSwift

/// Hyperbolic Tangent (Tanh) activation layer
/// Applies the tanh function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
/// Maps input values to range (-1, 1), providing zero-centered outputs
/// Often preferred over Sigmoid due to stronger gradients and zero-centered nature
public final class Tanh: BaseActivationLayer {
  /// Initializes a Tanh activation layer
  /// Tanh provides stronger gradients than Sigmoid and outputs are zero-centered
  /// - Parameter inputSize: Optional input tensor size. Required for first layer in network
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               type: .tanh,
               encodingType: .tanh)
  }
  
  /// Coding keys for serialization
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  /// Initializes Tanh layer from decoder for deserialization
  /// - Parameter decoder: Decoder containing serialized layer data
  /// - Throws: Decoding errors if deserialization fails
  convenience required public init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  /// Encodes the Tanh layer for serialization
  /// - Parameter encoder: Encoder to serialize layer data
  /// - Throws: Encoding errors if serialization fails
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }

  /// Called when input size is set, configures output size
  /// For Tanh, output size equals input size as it's an element-wise operation
  override public func onInputSizeSet() {
    outputSize = inputSize
  }
}
