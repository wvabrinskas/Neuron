//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/29/22.
//

import Foundation
import NumSwift

/// Rectified Linear Unit (ReLU) activation layer
/// Applies the element-wise ReLU function: f(x) = max(0, x)
/// ReLU is one of the most commonly used activation functions in deep learning
/// It helps mitigate the vanishing gradient problem and provides computational efficiency
public final class ReLu: BaseActivationLayer {
  /// Coding keys for serialization
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  /// Initializes a ReLU activation layer
  /// ReLU sets all negative values to zero while preserving positive values
  /// - Parameter inputSize: Optional input tensor size. Required for first layer in network
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               type: .reLu,
               encodingType: .relu)
  }
  
  /// Initializes ReLU layer from decoder for deserialization
  /// - Parameter decoder: Decoder containing serialized layer data
  /// - Throws: Decoding errors if deserialization fails
  convenience required public init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  /// Encodes the ReLU layer for serialization
  /// - Parameter encoder: Encoder to serialize layer data
  /// - Throws: Encoding errors if serialization fails
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(type, forKey: .type)
  }
  
  /// Called when input size is set, configures output size
  /// For ReLU, output size equals input size as it's an element-wise operation
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
  }
}
