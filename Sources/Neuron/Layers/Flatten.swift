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
  public init(inputSize: TensorSize? = nil) {
    super.init(inputSize: inputSize,
               biasEnabled: false,
               encodingType: .flatten)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
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
    let context = TensorContext { inputs, gradient, wrt in
      // Reshape gradient back to original inputSize by reinterpreting flat storage
      let inputSize = self.inputSize
      return (Tensor(storage: ContiguousArray(gradient.storage), size: inputSize), Tensor(), Tensor())
    }
    
    // Flatten: just reinterpret the flat storage as (total, 1, 1)
    let total = tensor.storage.count
    let flatSize = TensorSize(rows: 1, columns: total, depth: 1)
    let flat = Tensor(storage: ContiguousArray(tensor.storage), size: flatSize, context: context)
    
    flat.setGraph(tensor)
    
    return flat
  }
}
