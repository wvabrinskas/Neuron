//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import GameplayKit
import Numerics

public struct NormalDistribution {
  private let randomSource: GKRandomSource
  public let mean: Tensor.Scalar
  public let deviation: Tensor.Scalar
  
  public init(randomSource: GKRandomSource = GKRandomSource(), mean: Tensor.Scalar = 0, deviation: Tensor.Scalar = 0.01) {
    precondition(deviation >= 0)
    self.randomSource = randomSource
    self.mean = mean
    self.deviation = deviation
  }
  
  public func nextScalar() -> Tensor.Scalar {
    guard deviation > 0 else { return mean }
    
    let x1 = Tensor.Scalar(randomSource.nextUniform())
    let x2 = Tensor.Scalar(randomSource.nextUniform())
    let z1 = Tensor.Scalar.sqrt(-2 * Tensor.Scalar.log(x1)) * Tensor.Scalar.cos(2 * Tensor.Scalar.pi * x2)
    
    return z1 * deviation + mean
  }
  
  public func logProb(value: Tensor.Scalar) -> Tensor.Scalar {
    let loc = mean
    let variance = deviation * deviation
    let logScale = Tensor.Scalar.log(deviation)
    let otherPart = Tensor.Scalar.log(Tensor.Scalar.sqrt(2 * Tensor.Scalar.pi))
    return -(Tensor.Scalar.pow(value - loc, 2)) / (2 * variance) - logScale - otherPart
  }
}

public class Gaussian {
  // stored properties
  private var s : Double = 0.0
  private var v2 : Double = 0.0
  private var cachedNumberExists = false
  private var std: Double
  private var mean: Double
  
  public init(std: Double, mean: Double) {
    self.std = std
    self.mean = mean
  }
  
  public var gaussRand : Double  {
    var u1, u2, v1, x : Double
    if !cachedNumberExists {
      repeat {
        u1 = Double(arc4random()) / Double(UINT32_MAX)
        u2 = Double(arc4random()) / Double(UINT32_MAX)
        v1 = 2 * u1 - 1
        v2 = 2 * u2 - 1
        s = v1 * v1 + v2 * v2
      } while (s >= 1 || s == 0)
      x = v1 * sqrt(-2 * log(s) / s)
    }
    else {
      x = v2 * sqrt(-2 * log(s) / s)
    }
    cachedNumberExists = !cachedNumberExists
    x = x * std + mean
    return x
  }
}
