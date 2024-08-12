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
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               initializer: nil,
               biasEnabled: false,
               encodingType: .flatten)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type
  }
  
  override public func onInputSizeSet() {
    let total = inputSize.columns * inputSize.rows * inputSize.depth
    outputSize = TensorSize(array: [total, 1, 1])
  }
  
  convenience public required init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    
    let total = inputSize.columns * inputSize.rows * inputSize.depth
    self.outputSize = TensorSize(array: [total, 1, 1])
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
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
