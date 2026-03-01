//
//  Divide.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/28/26.
//


import Foundation
import NumSwift

public final class Divide: ArithmeticLayer {
  
  public init(inputSize: TensorSize = TensorSize(array: []),
              initializer: InitializerType = .heNormal,
              linkId: String = UUID().uuidString,
              inverse: Bool = false,
              linkTo: String) {
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: false,
               encodingType: .divide,
               inverse: inverse,
               linkId: linkId,
               linkTo: linkTo)
  }

  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let linkTo = try container.decodeIfPresent(String.self, forKey: .linkTo) ?? ""
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    self.init(linkId: linkId, linkTo: linkTo)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.linkTo = linkTo
  }
  
  override public func function(input: Tensor, other: Tensor) -> Tensor {
    if inverse {
      other / input
    } else {
      input / other
    }
  }
}
