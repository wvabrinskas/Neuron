//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/12/22.
//

import Foundation
import UIKit

public class BatchNormalizer {
  private var gamma: Float = 1
  private var beta: Float = 0
  private var movingMean: Float = 0
  private var movingVariance: Float = 0
  private var normalizedActivations: [Float] = []
  private var standardDeviation: Float = 0
  private let momentum: Float = 0.9

  func normalize(activations: [Float]) -> [Float] {
    
    let total = Float(activations.count)
    
    let mean = activations.reduce(0, +) / total
    
    let variance = activations.map { pow($0 - mean, 2) }.reduce(0, +) / total
        
    let e: Float = 0.00005 //this is a standard smoothing term
    let std = sqrt(variance + e)
      
    standardDeviation = std
  
    let normalized = activations.map { ($0 - mean) / std }
    
    normalizedActivations = normalized
    
    movingMean = momentum * movingMean + (1 - momentum) * mean
    movingVariance = momentum * movingVariance + (1 - movingVariance) * variance
    
    let normalizedScaledAndShifted = normalized.map { gamma * $0 + beta }
    
    return normalizedScaledAndShifted
  }
  
  func backward(gradient: [Float]) -> [Float] {
    let dBeta = gradient.reduce(0, +)
    
    guard gradient.count == normalizedActivations.count else {
      return gradient
    }
    
    //this isnt getting called =(
    var dGamma: Float = 0
    var outputGradients: [Float] = []
    
    let n: Float = Float(gradient.count)
    
    let dxNorm: [Float] = gradient.map { $0 * gamma }
    
    var dxNormTimesXNormSum: Float = 0
    
    for i in 0..<normalizedActivations.count {
      let xNormValue = normalizedActivations[i]
      let dxNormValue = dxNorm[i]
      dxNormTimesXNormSum += xNormValue * dxNormValue
    }
            
    let dxNormSum = dxNorm.reduce(0, +)
    let std = standardDeviation

    for gIndex in 0..<gradient.count {
      
      let xNorm = normalizedActivations[gIndex]
      let dxNorm = dxNorm[gIndex]
      
      let grad = gradient[gIndex]

      dGamma += grad * xNorm
          
      let dx = ((1 / n) / std) * (n * dxNorm - dxNormSum - xNorm * dxNormTimesXNormSum)
      outputGradients.append(dx)
    }
    
    gamma += dGamma
    beta += dBeta
    
    return outputGradients
  }
}
