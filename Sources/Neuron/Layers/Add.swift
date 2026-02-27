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

public final class Add: ArithmecticLayer {
  public init(inputSize: TensorSize = TensorSize(array: []),
              initializer: InitializerType = .heNormal,
              applyTo: EncodingType) {
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: false,
               encodingType: .add,
               applyTo: applyTo)
  }

  convenience public required init(from decoder: Decoder) throws {
    self.init(applyTo: .none)
    
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.applyTo = try container.decodeIfPresent(EncodingType.self, forKey: .applyTo) ?? .none
  }
  
  override public func function(input: Tensor, other: Tensor) -> Tensor {
    input + other
  }
}

