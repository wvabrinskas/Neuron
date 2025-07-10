//
//  File.swift
//
//
//  Created by William Vabrinskas on 4/29/22.
//

import Foundation
import NumSwift

/// Scaled Exponential Linear Unit (SELU) activation layer
/// Applies the SELU function with self-normalizing properties
/// f(x) = λ * x if x > 0, λ * α * (e^x - 1) if x ≤ 0
/// Designed to maintain zero mean and unit variance, enabling very deep networks
public final class SeLu: BaseActivationLayer {
  /// Coding keys for serialization
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  /// Initializes a SELU activation layer
  /// SELU has self-normalizing properties that can enable training of very deep networks
  /// - Parameter inputSize: Optional input tensor size. Required for first layer in network
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               type: .seLu,
               encodingType: .selu)
  }
  
  /// Initializes SeLu layer from decoder for deserialization
  /// - Parameter decoder: Decoder containing serialized layer data
  /// - Throws: Decoding errors if deserialization fails
  convenience required public init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  /// Encodes the SeLu layer for serialization
  /// - Parameter encoder: Encoder to serialize layer data
  /// - Throws: Encoding errors if serialization fails
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(type, forKey: .type)
  }
  
  /// Called when input size is set, configures output size
  /// For SeLu, output size equals input size as it's an element-wise operation
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
  }
}
