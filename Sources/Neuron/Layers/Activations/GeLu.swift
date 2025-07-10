//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/29/22.
//

import Foundation
import NumSwift

/// Gaussian Error Linear Unit (GELU) activation layer
/// Applies the GELU function: f(x) = x * Φ(x) where Φ is the cumulative distribution function
/// Approximation: f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
/// Commonly used in transformer models and modern architectures like BERT and GPT
public final class GeLu: BaseActivationLayer {
  /// Initializes a GELU activation layer
  /// GELU provides smooth activation with improved performance in transformer models
  /// - Parameter inputSize: Optional input tensor size. Required for first layer in network
  public init(inputSize: TensorSize = TensorSize(array: [])) {    
    super.init(inputSize: inputSize,
               type: .geLu,
               encodingType: .leakyRelu) // Note: May need to be updated to .gelu when available
  }
  
  /// Coding keys for serialization
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  /// Initializes GeLu layer from decoder for deserialization
  /// - Parameter decoder: Decoder containing serialized layer data
  /// - Throws: Decoding errors if deserialization fails
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.init()
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  /// Encodes the GeLu layer for serialization
  /// - Parameter encoder: Encoder to serialize layer data
  /// - Throws: Encoding errors if serialization fails
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(type, forKey: .type)
  }
  
  /// Called when input size is set, configures output size
  /// For GeLu, output size equals input size as it's an element-wise operation
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
  }
}
