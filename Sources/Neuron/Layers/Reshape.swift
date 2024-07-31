//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/4/22.
//

import Foundation
import NumSwift

/// Will take the inputSize as `[M * N * K, 1, 1]` and output a tensor of size `[M, N, K]`
public final class Reshape<N: TensorNumeric>: BaseLayer<N> {
  private let reshapeSize: TensorSize
  
  /// Default initializer for a reshape layer.
  /// - Parameters:
  ///   - size: The size to reshape to.
  ///   - inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(to size: TensorSize, inputSize: TensorSize = TensorSize(array: [])) {
    reshapeSize = size
    super.init(inputSize: inputSize,
               initializer: nil,
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
    self.weights = try container.decodeIfPresent(Tensor<N>.self, forKey: .weights) ?? Tensor<N>()
    self.biases = try container.decodeIfPresent(Tensor<N>.self, forKey: .biases) ?? Tensor<N>()
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
  
  public override func forward(tensor: Tensor<N>) -> Tensor<N> {
    let context = TensorContext<N> { inputs, gradient in
      let value: [Tensor<N>.Scalar] = gradient.value.flatten()
      return (Tensor<N>(value), Tensor<N>(), Tensor<N>())
    }
    
    let flat: [Tensor<N>.Scalar] = tensor.value.flatten()
    
    let sizeCols = reshapeSize.columns
    let sizeRows = reshapeSize.rows
    
    let reshaped = flat.reshape(columns: sizeCols).reshape(columns: sizeRows)
    
    let out = Tensor<N>(reshaped, context: context)
    
    out.setGraph(tensor)

    return out
  }
  
  override public func onInputSizeSet() {
    outputSize = reshapeSize
  }
}
