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
  /// Default initializer for a leaky relu activation function.
  /// - Parameter limit: The alpha limit value for leaky relu.
  public init() {    
    super.init(type: .geLu,
               encodingType: .leakyRelu)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type,
         limit
  }
  
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.init()
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(type, forKey: .type)
  }
  
  override public func onInputSizeSet() {
    outputSize = inputSize
  }
}
