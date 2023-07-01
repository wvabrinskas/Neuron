//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/4/22.
//

import Foundation
import NumSwift

/// Will take the inputSize as `[M * N * K, 1, 1]` and output a tensor of size `[M, N, K]`
public final class Reshape: Layer {
  public var encodingType: EncodingType = .reshape
  public var device: Device = CPU()
  public var biasEnabled: Bool = true
  public var isTraining: Bool = true

  public var inputSize: TensorSize = TensorSize(array: [])
  public var outputSize: TensorSize {
    reshapeSize
  }
  
  public var weights: Tensor = Tensor()
  public var biases: Tensor = Tensor()
  public var trainable: Bool = true
  public var initializer: Initializer?
  private let reshapeSize: TensorSize
  
  /// Default initializer for a reshape layer.
  /// - Parameters:
  ///   - size: The size to reshape to.
  ///   - inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(to size: TensorSize, inputSize: TensorSize = TensorSize(array: [])) {
    reshapeSize = size
    self.inputSize = inputSize
  }
  
  enum CodingKeys: String, CodingKey {
    case biasEnabled,
         inputSize,
         weights,
         biases,
         reshapeSize,
         type
  }
  
  convenience public init(from decoder: Decoder) throws {
    self.init(to: TensorSize(array: []))
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.weights = try container.decodeIfPresent(Tensor.self, forKey: .weights) ?? Tensor()
    self.biases = try container.decodeIfPresent(Tensor.self, forKey: .biases) ?? Tensor()
    let resize = try container.decodeIfPresent(TensorSize.self, forKey: .reshapeSize) ?? TensorSize(array: [])
    self.init(to: resize)
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(weights, forKey: .weights)
    try container.encode(biases, forKey: .biases)
    try container.encode(reshapeSize, forKey: .reshapeSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  public func forward(tensor: Tensor) -> Tensor {
    let context = TensorContext { inputs, gradient in
      let value: [Tensor.Scalar] = gradient.value.flatten()
      return (Tensor(value), Tensor())
    }
    
    let flat: [Tensor.Scalar] = tensor.value.flatten()
    
    let sizeCols = reshapeSize.columns
    let sizeRows = reshapeSize.rows
    
    let reshaped = flat.reshape(columns: sizeCols).reshape(columns: sizeRows)
    
    let out = Tensor(reshaped, context: context)
    
    out.setGraph(tensor)

    return out
  }
  
  public func apply(gradients: Optimizer.Gradient, learningRate: Float) {
    //no opp
  }
  
}
