//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/12/22.
//

import Foundation
import NumSwift

public class BatchNormalizer {
  @TestNaN public var gamma: Float = 1
  @TestNaN public var beta: Float = 0
  @TestNaN public var movingMean: Float = 1
  @TestNaN public var movingVariance: Float = 1
  private var normalizedActivations: [Float] = []
  private var standardDeviation: Float = 0
  private let e: Float = 0.00005 //this is a standard smoothing term
  private let learningRate: Float
  
  public init(gamma: Float = 1,
              beta: Float = 0,
              learningRate: Float,
              movingMean: Float = 1,
              movingVariance: Float = 1) {
    self.gamma = gamma
    self.beta = beta
    self.learningRate = learningRate
    self.movingVariance = movingVariance
    self.movingMean = movingMean
  }

  public func normalize(activations: [Float], training: Bool) -> [Float] {
    
    //gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon) + beta.
    
    let total = Float(activations.count)
    
    let mean = training == true ? activations.sum / total : movingMean
    
    let diffVar = activations - mean
    let variance = training == true ? diffVar.sumOfSquares / total : movingVariance
            
    let std = sqrt(variance + e)
      
    standardDeviation = std
  
    let normalized = activations.map { ($0 - mean) / std }
    
    normalizedActivations = normalized
    
    if training {
      movingMean *= learningRate + mean * (1 - learningRate)
      movingVariance *= learningRate + variance * (1 - learningRate)
    }
        
    let adjustedWithGamma = normalized * gamma
    let normalizedScaledAndShifted = adjustedWithGamma + beta
    
    return normalizedScaledAndShifted
  }
  
  public func backward(gradient: [Float]) -> [Float] {
    let dBeta = gradient.sum
    
    guard gradient.count == normalizedActivations.count else {
      return gradient
    }
    
    var outputGradients: [Float] = []
    
    let n: Float = Float(gradient.count)
    
    let dxNorm: [Float] = gradient * gamma
    
    let combineXandDxNorm = normalizedActivations * dxNorm
    let dxNormTimesXNormSum: Float = combineXandDxNorm.sum
 
    let dxNormSum = dxNorm.sum
    let std = standardDeviation
    
    let dGamma = (gradient * normalizedActivations).sum
    
    let t = dxNorm * n
    let dx = (t - (dxNormSum - normalizedActivations) * dxNormTimesXNormSum) * ((1 / n) / std)
      
    outputGradients = dx
    
    gamma -= learningRate * dGamma
    beta -= learningRate * dBeta
    
    return outputGradients
  }
}
