//
//  BetaDistribution.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/11/26.
//

import Foundation
import Numerics

/// A distribution that models random variables constrained to the interval [0, 1],
/// parameterized by two positive shape parameters alpha and beta.
public struct BetaDistribution {
  
  /// Samples a random scalar from a Beta(`alpha`, `beta`) distribution.
  ///
  /// - Parameters:
  ///   - alpha: First Beta shape parameter.
  ///   - beta: Second Beta shape parameter.
  /// - Returns: Random value in `[0, 1]`.
  public static func randomBeta(_ alpha: Tensor.Scalar, _ beta: Tensor.Scalar) -> Tensor.Scalar {
    let x = randomGamma(shape: alpha)
    let y = randomGamma(shape: beta)
    return x / (x + y)
  }
  
  // Marsaglia and Tsang's method for Gamma distribution
  private static func randomGamma(shape: Tensor.Scalar) -> Tensor.Scalar {
    if shape < 1.0 {
      return randomGamma(shape: shape + 1.0) * Tensor.Scalar.pow(Tensor.Scalar.random(in: 0..<1), 1.0 / shape)
    }
    
    let d = shape - 1.0 / 3.0
    let c = 1.0 / Tensor.Scalar.sqrt((9.0 * d) + Tensor.Scalar.stabilityFactor)
    
    while true {
      var x: Tensor.Scalar
      var v: Tensor.Scalar
      repeat {
        // Box-Muller for normal sample
        let u1 = Tensor.Scalar.random(in: 0..<1)
        let u2 = Tensor.Scalar.random(in: 0..<1)
        x = Tensor.Scalar.sqrt(-2.0 * Tensor.Scalar.log(u1)) * Tensor.Scalar.cos(2.0 * .pi * u2)
        v = 1.0 + c * x
      } while v <= 0
      
      v = v * v * v
      let u = Tensor.Scalar.random(in: 0..<1)
      
      if u < 1.0 - 0.0331 * (x * x) * (x * x) {
        return d * v
      }
      if Tensor.Scalar.log(u) < 0.5 * x * x + d * (1.0 - v + Tensor.Scalar.log(v)) {
        return d * v
      }
    }
  }

}
