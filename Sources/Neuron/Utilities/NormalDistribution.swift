//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import GameplayKit
import Numerics

/// A sampler that generates random values from a normal (Gaussian) distribution.
public struct NormalDistribution {
  private let randomSource: GKRandomSource
  /// The mean (expected value) of the normal distribution.
  public let mean: Tensor.Scalar
  /// The standard deviation of the normal distribution.
  public let deviation: Tensor.Scalar
  
  /// Creates a normal distribution sampler.
  ///
  /// - Parameters:
  ///   - randomSource: Random source used for sampling.
  ///   - mean: Distribution mean.
  ///   - deviation: Standard deviation (must be non-negative).
  public init(randomSource: GKRandomSource = GKRandomSource(), mean: Tensor.Scalar = 0, deviation: Tensor.Scalar = 0.01) {
    precondition(deviation >= 0)
    self.randomSource = randomSource
    self.mean = mean
    self.deviation = deviation
  }
  
  /// Samples the next scalar value from the distribution.
  ///
  /// - Returns: Random scalar drawn from `N(mean, deviation^2)`.
  public func nextScalar() -> Tensor.Scalar {
    guard deviation > 0 else { return mean }
    
    let x1 = Tensor.Scalar(randomSource.nextUniform())
    let x2 = Tensor.Scalar(randomSource.nextUniform())
    let z1 = Tensor.Scalar.sqrt(-2 * Tensor.Scalar.log(x1)) * Tensor.Scalar.cos(2 * Tensor.Scalar.pi * x2)
    
    return z1 * deviation + mean
  }
  
  /// Computes the log probability density for a given value.
  ///
  /// - Parameter value: Scalar for which to evaluate log-density.
  /// - Returns: Log-probability under this normal distribution.
  public func logProb(value: Tensor.Scalar) -> Tensor.Scalar {
    let loc = mean
    let variance = deviation * deviation
    let logScale = Tensor.Scalar.log(deviation)
    let otherPart = Tensor.Scalar.log(Tensor.Scalar.sqrt(2 * Tensor.Scalar.pi))
    return -(Tensor.Scalar.pow(value - loc, 2)) / (2 * variance) - logScale - otherPart
  }
}

/// A Gaussian random number generator using the Box-Muller transform.
public class Gaussian {
  // stored properties
  private var s : Double = 0.0
  private var v2 : Double = 0.0
  private var cachedNumberExists = false
  private var std: Double
  private var mean: Double
  
  /// Creates a Box-Muller Gaussian sampler.
  ///
  /// - Parameters:
  ///   - std: Standard deviation.
  ///   - mean: Distribution mean.
  public init(std: Double, mean: Double) {
    self.std = std
    self.mean = mean
  }
  
  /// A randomly sampled value from the Gaussian distribution using the Box-Muller transform.
  ///
  /// - Returns: A `Double` sampled from the distribution with the configured mean and standard deviation.
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
