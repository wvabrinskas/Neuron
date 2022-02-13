//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/28/21.
//

import Foundation

public enum Optimizer: Codable {
  case adam(b1: Float = 0.9,
            b2: Float = 0.999,
            eps: Float = 1e-8)
  
  func get(learningRate: Float) -> OptimizerFunction {
    switch self {
    case let .adam(b1, b2, eps):
      return Adam(b1: b1, b2: b2, eps: eps, alpha: learningRate)
    }
  }
}

public protocol OptimizerFunction: AnyObject {
  func run(weight: Float, gradient: Float) -> Float
  func runBias(weight: Float, gradient: Float) -> Float
  func step()
  func zero()
}

public class Adam: OptimizerFunction {

  private var b1: Float = 0.9
  private var b2: Float = 0.999
  private var eps: Float = 1e-8
  private var alpha: Float
  
  private var m: Float = 0
  private var v: Float = 0
  private var mb: Float = 0
  private var vb: Float = 0
  private var t: Float = 0
  
  public init(b1: Float = 0.9,
              b2: Float = 0.999,
              eps: Float = 1e-8,
              alpha: Float) {
    
    self.b1 = b1
    self.b2 = b2
    self.eps = eps
    self.alpha = alpha
  }
  
  public func runBias(weight: Float, gradient: Float) -> Float {
    mb = b1 * mb + (1 - b1) * gradient
    vb = b2 * vb + (1 - b2) * pow(gradient, 2)
    let mHat = mb / (1 - pow(b1, Float(t)))
    let vHat = vb / (1 - pow(b2, Float(t)))
    let newW = weight - alpha * mHat / (sqrt(vHat) + eps) * gradient
    return newW
  }
  
  public func run(weight: Float, gradient: Float) -> Float {
    m = b1 * m + (1 - b1) * gradient
    v = b2 * v + (1 - b2) * pow(gradient, 2)
    let mHat = m / (1 - pow(b1, Float(t)))
    let vHat = v / (1 - pow(b2, Float(t)))
    let newW = weight - alpha * (mHat / (sqrt(vHat) + eps)) * gradient
    return newW
  }
  
  public func zero() {
    self.t = 0
  }
  
  public func step() {
    self.t += 1
  }
}
