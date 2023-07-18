//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/29/22.
//

import Foundation
import NumSwift

/// Performs a LeakyRelu activation.
public final class LeakyReLu: ActivationLayer {
  public var encodingType: EncodingType = .leakyRelu
  public var device: Device = CPU()
  public var biasEnabled: Bool = true
  public var trainable: Bool = true
  public var type: Activation
  public var inputSize: TensorSize = TensorSize(array: []) {
    didSet {
      outputSize = inputSize
    }
  }
  public var outputSize: TensorSize = TensorSize(array: [])
  public var weights: Tensor = Tensor()
  public var biases: Tensor = Tensor()
  public var initializer: Initializer?
  public var isTraining: Bool = true

  private var limit: Float
  
  /// Default initializer for a leaky relu activation function.
  /// - Parameter limit: The alpha limit value for leaky relu.
  public init(limit: Float = 0.01) {
    type = .leakyRelu(limit: limit)
    self.limit = limit
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type,
         limit
  }
  
  convenience public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let limit = try container.decodeIfPresent(Float.self, forKey: .limit) ?? 0.01
    self.init(limit: limit)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(type, forKey: .type)
    try container.encode(limit, forKey: .limit)
  }
  
  public func forward(tensor: Tensor) -> Tensor {
    
    let context = TensorContext { inputs, gradient in
      let out = self.device.derivate(inputs, self.type).value * gradient.value
      return (Tensor(out), Tensor(), Tensor())
    }
    
    let result = device.activate(tensor, type)
    let out = Tensor(result.value, context: context)
    out.label = type.asString()
    
    out.setGraph(tensor)
    return out
  }
  
  public func apply(gradients: Optimizer.Gradient, learningRate: Float) {
    //no op
  }
}
