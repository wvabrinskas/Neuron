//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/29/22.
//

import Foundation
import NumSwift

/// Performs a Sigmoid activation.
public final class Sigmoid: ActivationLayer {
  public var encodingType: EncodingType = .sigmoid
  public var device: Device = CPU()
  public var biasEnabled: Bool = true
  public var trainable: Bool = true
  public var type: Activation = .sigmoid
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

  /// Default initializer for a Sigmoid activation.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    self.inputSize = inputSize
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  convenience public init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
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

