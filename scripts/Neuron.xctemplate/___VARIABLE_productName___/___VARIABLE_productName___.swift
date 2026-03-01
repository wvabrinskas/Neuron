import Foundation
import NumSwift

//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/4/22.
//

import Foundation
import NumSwift

public final class ___VARIABLE_productName___: BaseLayer {
  public init(inputSize: TensorSize = TensorSize(array: []),
              initializer: InitializerType = .heNormal,
              linkId: String = UUID().uuidString) {
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: false,  
               linkId: linkId,
               encodingType: .___VARIABLE_encodingType___)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type, linkId
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    /// do something when the input size is set when calling `compile` on `Sequential`
    /// like setting the output size or initializing the weights
  }
  
  convenience public required init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(linkId, forKey: .linkId)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    let context = TensorContext { inputs, gradient, wrt in
      // backpropogation calculation
      return (Tensor(), Tensor(), Tensor())
    }
    
    // forward calculation
    let out = Tensor()
    return super.forward(tensor: out, context: context)
  }
}

