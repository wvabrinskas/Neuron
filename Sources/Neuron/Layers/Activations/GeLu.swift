//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/29/22.
//

import Foundation
import NumSwift

/// Performs a GeLu activation.
public final class GeLu: BaseActivationLayer {
  /// Creates a GeLU activation layer.
  public init(linkId: String = UUID().uuidString) {
    super.init(type: .geLu,
               linkId: linkId,
               encodingType: .leakyRelu)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type,
         limit,
         linkId
  }
  
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    self.init(linkId: linkId)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  /// Encodes GeLU layer configuration.
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
