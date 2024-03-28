//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/2/22.
//

import Foundation
import NumSwift

/// Performs a tanh activation.
public final class Tanh: BaseActivationLayer {
  /// Default initializer for a Tanh activation.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               type: .tanh,
               encodingType: .tanh)
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

  override public func onInputSizeSet() {
    outputSize = inputSize
  }
}
