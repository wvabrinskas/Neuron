//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/4/22.
//

import Foundation
import NumSwift

/// Will take an inputSize of [M, N, K] and outputs [M * N * K, 1, 1]
public final class Flatten: Layer {
  public var encodingType: EncodingType = .flatten
  public var device: Device = CPU()
  public var biasEnabled: Bool = true
  public var inputSize: TensorSize = TensorSize(array: []) {
    didSet {
      let total = inputSize.columns * inputSize.rows * inputSize.depth
      outputSize = TensorSize(array: [total, 1, 1])
    }
  }
  
  public var outputSize: TensorSize = TensorSize(array: [])
  public var weights: Tensor = Tensor()
  public var biases: Tensor = Tensor()
  public var trainable: Bool = true
  public var initializer: Initializer?
  public var isTraining: Bool = true

  /// Default initializer for Flatten layer.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    self.inputSize = inputSize
    
    let total = inputSize.columns * inputSize.rows * inputSize.depth
    outputSize = TensorSize(array: [total, 1, 1])
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type
  }
  
  convenience public init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    
    let total = inputSize.columns * inputSize.rows * inputSize.depth
    self.outputSize = TensorSize(array: [total, 1, 1])
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  public func forward(tensor: Tensor) -> Tensor {
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
  
  public func apply(gradients: Optimizer.Gradient, learningRate: Float) {
   //no op
  }
}
