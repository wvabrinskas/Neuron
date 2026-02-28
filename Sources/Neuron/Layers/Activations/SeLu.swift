//
//  File.swift
//
//
//  Created by William Vabrinskas on 4/29/22.
//

import Foundation
import NumSwift

/// Performs a Selu activation.
public final class SeLu: BaseActivationLayer {
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type,
         linkId
  }
  
  /// Default initializer for a Relu activation.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize? = nil,
              linkId: String = UUID().uuidString) {
    super.init(inputSize: inputSize,
               type: .seLu,
               linkId: linkId,
               encodingType: .selu)
  }
  
  convenience required public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    self.init(linkId: linkId)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  /// Encodes SELU layer configuration.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(type, forKey: .type)
    try container.encode(linkId, forKey: .linkId)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
  }
}
