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
  public let learningRate: Float
  public let momentum: Float
  
  private var normalizedActivations: [Float] = []
  private var standardDeviation: Float = 0
  private let e: Float = 0.00005 //this is a standard smoothing term
  private var ivar: Float = 0
  private var dGamma: Float = 0
  private var dBeta: Float = 0
  
  public init(gamma: Float = 1,
              beta: Float = 0,
              momentum: Float,
              learningRate: Float,
              movingMean: Float = 1,
              movingVariance: Float = 1) {
    self.gamma = gamma
    self.beta = beta
    self.learningRate = learningRate
    self.movingVariance = movingVariance
    self.movingMean = movingMean
    self.momentum = momentum
  }

  public func normalize(activations: [Float], training: Bool) -> [Float] {
        
    let total = Float(activations.count)
    
    let mean = training == true ? activations.sum / total : movingMean
    
    let diffVar = activations - mean

    let variance = training == true ? diffVar.sumOfSquares / total : movingVariance
            
    let std = sqrt(variance + e)
    
    ivar = 1 / std
      
    standardDeviation = std
  
    let normalized = diffVar * ivar
    
    normalizedActivations = normalized
    
    if training {
      movingMean = momentum * movingMean + (1 - momentum) * mean
      movingVariance = momentum * movingVariance + (1 - momentum) * variance
    }
        
    let normalizedScaledAndShifted = gamma * normalized + beta
    
    return normalizedScaledAndShifted
  }
  
  public func backward(gradient: [Float]) -> [Float] {
    guard gradient.count == normalizedActivations.count else {
      return gradient
    }

    var outputGradients: [Float] = []
    
    let n: Float = Float(gradient.count)
    
    let dxHat: [Float] = gradient * gamma
    
    let firstTerm = (1 / n) * ivar
    
    let secondTerm = n * dxHat
    
    let thirdTerm = dxHat.sum
    
    let fourthTerm = normalizedActivations * (normalizedActivations * dxHat).sum
    
    let dx = firstTerm * (secondTerm - thirdTerm - fourthTerm)
      
    outputGradients = dx
    
    dGamma += (gradient * normalizedActivations).sum
    dBeta += gradient.sum

    return outputGradients
  }
  
  public func adjustLearnables(batchSize: Int) {
    gamma -= learningRate * (dGamma / Float(batchSize))
    beta -= learningRate * (dBeta / Float(batchSize))
  }
}
