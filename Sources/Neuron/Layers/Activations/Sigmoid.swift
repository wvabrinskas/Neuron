//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/29/22.
//

import Foundation
import NumSwift

/// Sigmoid activation layer
/// Applies the sigmoid function: f(x) = 1 / (1 + e^(-x))
/// Maps input values to range (0, 1), making it useful for binary classification
/// Has saturating gradients which can lead to vanishing gradient problems in deep networks
public final class Sigmoid: BaseActivationLayer {
  /// Initializes a Sigmoid activation layer
  /// Sigmoid function smoothly maps values to probabilities between 0 and 1
  /// - Parameter inputSize: Optional input tensor size. Required for first layer in network
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               type: .sigmoid,
               encodingType: .sigmoid)
  }
  
  /// Coding keys for serialization
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  /// Initializes Sigmoid layer from decoder for deserialization
  /// - Parameter decoder: Decoder containing serialized layer data
  /// - Throws: Decoding errors if deserialization fails
  convenience required public init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  /// Encodes the Sigmoid layer for serialization
  /// - Parameter encoder: Encoder to serialize layer data
  /// - Throws: Encoding errors if serialization fails
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  /// Called when input size is set, configures output size
  /// For Sigmoid, output size equals input size as it's an element-wise operation
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
  }
}

