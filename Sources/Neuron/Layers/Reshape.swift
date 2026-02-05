//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/4/22.
//

import Foundation
import NumSwift

/// Will take the inputSize as `[M * N * K, 1, 1]` and output a tensor of size `[M, N, K]`
public final class Reshape: BaseLayer {
  private let reshapeSize: TensorSize
  
  /// Default initializer for a reshape layer.
  /// - Parameters:
  ///   - size: The size to reshape to.
  ///   - inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(to size: TensorSize, inputSize: TensorSize? = nil) {
    reshapeSize = size
    super.init(inputSize: inputSize,
               biasEnabled: false,
               encodingType: .reshape)
  }
  
  enum CodingKeys: String, CodingKey {
    case biasEnabled,
         inputSize,
         weights,
         biases,
         reshapeSize,
         type
  }
  
  convenience public required init(from decoder: Decoder) throws {
    self.init(to: TensorSize(array: []))
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.weights = try container.decodeIfPresent(Tensor.self, forKey: .weights) ?? Tensor()
    self.biases = try container.decodeIfPresent(Tensor.self, forKey: .biases) ?? Tensor()
    let resize = try container.decodeIfPresent(TensorSize.self, forKey: .reshapeSize) ?? TensorSize(array: [])
    self.init(to: resize)
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(weights, forKey: .weights)
    try container.encode(biases, forKey: .biases)
    try container.encode(reshapeSize, forKey: .reshapeSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let context = TensorContext { inputs, gradient, wrt in
      // Reshape gradient back to flat (the input was flattened)
      let flatSize = TensorSize(rows: 1, columns: gradient.storage.count, depth: 1)
      return (Tensor(storage: ContiguousArray(gradient.storage), size: flatSize), Tensor(), Tensor())
    }
    
    // Reshape: reinterpret the flat storage with the new shape
    let out = Tensor(storage: ContiguousArray(tensor.storage), size: reshapeSize, context: context)
    
    out.setGraph(tensor)

    return out
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = reshapeSize
  }
}
