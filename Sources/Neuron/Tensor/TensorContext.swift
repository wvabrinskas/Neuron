//
//  File.swift
//  
//
//  Created by William Vabrinskas on 7/31/22.
//

import Foundation

public struct TensorContext<N: TensorNumeric>: Codable {
  public typealias TensorBackpropResult = (input: Tensor<N>, weight: Tensor<N>, bias: Tensor<N>)
  public typealias TensorContextFunction = (_ inputs: Tensor<N>, _ gradient: Tensor<N>) -> TensorBackpropResult
  var backpropagate: TensorContextFunction
  
  public init(backpropagate: TensorContextFunction? = nil) {
    let defaultFunction = { (input: Tensor<N>, gradient: Tensor<N>) in
      return (Tensor<N>(gradient.value), Tensor<N>(), Tensor<N>())
    }
    
    self.backpropagate = backpropagate ?? defaultFunction
  }
  
  public func encode(to encoder: Encoder) throws {}
  
  public init(from decoder: Decoder) throws {
    self = TensorContext<N>()
  }
}
