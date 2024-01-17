//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/2/22.
//

import Foundation
import NumSwift

/// Performs a Swish activation.
public final class Swish: BaseActivationLayer {

  /// Default initializer for a Swish activation.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               type: .swish,
               encodingType: .swish)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type
  }
  
  convenience required public init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }

  public override func forward(tensor: Tensor) -> Tensor {
    
    let context = TensorContext { inputs, gradient in
      let out = self.device.derivate(inputs, self.type, inputSize: self.inputSize).value * gradient.value
      return (Tensor(out), Tensor(), Tensor())
    }
    
    let result = device.activate(tensor, type, inputSize: inputSize)
    let out = Tensor(result.value, context: context)
    out.label = type.asString()

    out.setGraph(tensor)

    return out
  }
  
  override public func onInputSizeSet() {
    outputSize = inputSize
  }
  
}

