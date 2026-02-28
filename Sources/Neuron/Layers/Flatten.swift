//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/4/22.
//

import Foundation
import NumSwift

/// Will take an inputSize of [M, N, K] and outputs [M * N * K, 1, 1]
public final class Flatten: BaseLayer {
  /// Default initializer for Flatten layer.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize? = nil,
              linkId: String = UUID().uuidString) {
    super.init(inputSize: inputSize,
               biasEnabled: false,
               linkId: linkId,
               encodingType: .flatten)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type, linkId
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    let total = inputSize.columns * inputSize.rows * inputSize.depth
    outputSize = TensorSize(array: [total, 1, 1])
  }
  
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    self.init(linkId: linkId)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    
    let total = inputSize.columns * inputSize.rows * inputSize.depth
    self.outputSize = TensorSize(array: [total, 1, 1])
  }
  
  /// Encodes flatten layer shape metadata.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(linkId, forKey: .linkId)
  }
  
  /// Flattens the input tensor into shape `[columns*rows*depth, 1, 1]`.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Flattened tensor with reshape-aware backpropagation context.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let tensorContext = TensorContext { inputs, gradient, wrt in
      // Reshape gradient back to original inputSize by reinterpreting flat storage
      let inputSize = self.inputSize
      return (Tensor(gradient.storage, size: inputSize), Tensor(), Tensor())
    }
    
    // Flatten: just reinterpret the flat storage as (total, 1, 1)
    let total = tensor.storage.count
    let flatSize = TensorSize(rows: 1, columns: total, depth: 1)
    let flat = Tensor(tensor.storage, size: flatSize, context: tensorContext)
    
    flat.setGraph(tensor)
    
    return super.forward(tensor: flat, context: context)
  }
}
