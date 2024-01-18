//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/29/22.
//

import Foundation
import NumSwift

/// Performs a LeakyRelu activation.
public final class LeakyReLu: BaseActivationLayer {
  private var limit: Float
  
  /// Default initializer for a leaky relu activation function.
  /// - Parameter limit: The alpha limit value for leaky relu.
  public init(limit: Float = 0.01) {
    self.limit = limit
    
    super.init(type: .leakyRelu(limit: limit),
               encodingType: .leakyRelu)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type,
         limit
  }
  
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let limit = try container.decodeIfPresent(Float.self, forKey: .limit) ?? 0.01
    self.init(limit: limit)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(type, forKey: .type)
    try container.encode(limit, forKey: .limit)
  }
  
  public override func forward(tensor: Tensor) -> Tensor {
    
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
  
  override public func onInputSizeSet() {
    outputSize = inputSize
  }
}
