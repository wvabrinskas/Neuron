//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/28/21.
//

import Foundation

public enum Optimizer {
  case adam(b1: Float = 0.9,
            b2: Float = 0.999,
            eps: Float = 1e-8)
  
  func get() -> OptimizerFunction {
    switch self {
    case let .adam(b1, b2, eps):
      return Adam(b1: b1, b2: b2, eps: eps)
    }
  }
}

public protocol OptimizerFunction: class {
  func run(alpha: Float, weight: Float, gradient: Float) -> Float
}

public class Adam: OptimizerFunction {

  private var b1: Float = 0.9
  private var b2: Float = 0.999
  private var eps: Float = 1e-8
  
  private var m: Float = 0
  private var v: Float = 0
  private var t: Float = 1
  
  public init(b1: Float = 0.9,
              b2: Float = 0.999,
              eps: Float = 1e-8) {
    
    self.b1 = b1
    self.b2 = b2
    self.eps = eps
  }
  
  public func run(alpha: Float, weight: Float, gradient: Float) -> Float {
    m = b1 * m + (1 - b1) * gradient
    v = b2 * v + (1 - b2) * pow(gradient, 2)
    let mHat = m / (1 - pow(b1, Float(t)))
    let vHat = v / (1 - pow(b2, Float(t)))
    let newW = weight - alpha * mHat / (sqrt(vHat) + eps)
    t += 1
    return newW
  }
}
