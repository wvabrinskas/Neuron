//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/29/22.
//

import Foundation
import NumSwift

/// Performs a Relu activation.
public final class ReLu: ActivationLayer {
  public var encodingType: EncodingType = .relu
  public var device: Device = CPU()
  public var biasEnabled: Bool = true
  public var trainable: Bool = true
  public var type: Activation = .reLu
  public var inputSize: TensorSize = TensorSize(array: []) {
    didSet {
      outputSize = inputSize
    }
  }
  public var outputSize: TensorSize = TensorSize(array: [])
  public var weights: Tensor = Tensor()
  public var biases: Tensor = Tensor()
  public var initializer: Initializer?
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  /// Default initializer for a Relu activation.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    self.inputSize = inputSize
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
    try container.encode(type, forKey: .type)
  }

  public func forward(tensor: Tensor) -> Tensor {
    
    let context = TensorContext { inputs, gradient in
      let derivate = self.device.derivate(inputs, self.type, inputSize: self.inputSize)
      let out = derivate.value * gradient.value
      return (Tensor(out), Tensor())
    }
    
    let result = device.activate(tensor, type, inputSize: inputSize)
    let out = Tensor(result.value, context: context)
    out.label = type.asString()

    return out
  }
  
  public func apply(gradients: Optimizer.Gradient) {
    //no op
  }
}
