//
//  File.swift
//  
//
//  Created by William Vabrinskas on 7/31/22.
//

import Foundation

public struct TensorContext: Codable {
  public typealias TensorBackpropResult = (input: Tensor, weight: Tensor, bias: Tensor)
  public typealias TensorContextFunction = (_ inputs: Tensor, _ gradient: Tensor, _ wrt: Tensor?) -> TensorBackpropResult
  var backpropagate: TensorContextFunction
  
  public init(backpropagate: TensorContextFunction? = nil) {
    let defaultFunction = { (input: Tensor, gradient: Tensor, wrt: Tensor?) in
      return (Tensor(gradient.value), Tensor(), Tensor())
    }
    
    self.backpropagate = backpropagate ?? defaultFunction
  }
  
  public func encode(to encoder: Encoder) throws {}
  
  public init(from decoder: Decoder) throws {
    self = TensorContext()
  }
}
