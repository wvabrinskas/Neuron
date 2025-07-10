//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/4/22.
//

import Foundation
import NumSwift

/// Flatten layer for converting multi-dimensional tensors to 1D
/// Transforms input tensor from [M, N, K] to [M * N * K, 1, 1]
/// Commonly used between convolutional and fully connected layers
/// Essential for transitioning from spatial features to dense layers
public final class Flatten: BaseLayer {
  /// Initializes a Flatten layer
  /// Converts multi-dimensional input to a single dimension
  /// - Parameter inputSize: Input tensor dimensions. Required for first layer in network
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               initializer: nil,
               biasEnabled: false,
               encodingType: .flatten)
  }
  
  /// Coding keys for serialization
  enum CodingKeys: String, CodingKey {
    case inputSize, type
  }
  
  /// Called when input size is set, calculates flattened output dimensions
  /// Total output size is the product of all input dimensions
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    let total = inputSize.columns * inputSize.rows * inputSize.depth
    outputSize = TensorSize(array: [total, 1, 1])
  }
  
  /// Initializes Flatten layer from decoder for deserialization
  /// - Parameter decoder: Decoder containing serialized layer data
  /// - Throws: Decoding errors if deserialization fails
  convenience public required init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    
    let total = inputSize.columns * inputSize.rows * inputSize.depth
    self.outputSize = TensorSize(array: [total, 1, 1])
  }
  
  /// Encodes the Flatten layer for serialization
  /// - Parameter encoder: Encoder to serialize layer data
  /// - Throws: Encoding errors if serialization fails
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  /// Performs forward pass through the flatten layer
  /// Converts multi-dimensional input to 1D while preserving gradient flow
  /// - Parameters:
  ///   - tensor: Multi-dimensional input tensor to flatten
  ///   - context: Network context for computation
  /// - Returns: Flattened 1D tensor with proper gradient context
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let context = TensorContext { inputs, gradient in
      
      let inputSize = self.inputSize
      let deltas: [Tensor.Scalar] = gradient.value.flatten()
      
      let batchedDeltas = deltas.batched(into: inputSize.columns * inputSize.rows)
      let gradients = batchedDeltas.map { $0.reshape(columns: inputSize.columns) }
      
      return (Tensor(gradients), Tensor(), Tensor())
    }
    
    let flatten: [Tensor.Scalar] = tensor.value.flatten()
    let flat = Tensor(flatten, context: context)
    
    return flat
  }
}
