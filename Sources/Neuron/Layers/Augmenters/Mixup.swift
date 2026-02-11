//
//  Mixup.swift
//
//

import Foundation
import NumSwift

public final class Mixup: BaseLayer {
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               biasEnabled: false,
               encodingType: .mixup)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type
  }
  
  override public func onInputSizeSet() {
    /// do something when the input size is set when calling `compile` on `Sequential`
  }
  
  convenience public required init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    let tensorContext = TensorContext { inputs, gradient, wrt in
      // backpropogation calculation
      return (Tensor(), Tensor(), Tensor())
    }
    
    // forward calculation
    return Tensor()
  }
}

