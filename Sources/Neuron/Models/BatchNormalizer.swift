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
  private var centeredActivations: [Float] = []
  private var standardDeviations: [Float] = []

  func normalize(activations: [Float]) -> [Float] {
    
    let total = Float(activations.count)
    
    movingMean = activations.reduce(0, +) / total
    
    movingVariance = activations.map { activation in
      
      let centered = activation - movingMean
      centeredActivations.append(centered)
      
      return pow(centered, 2)
      
    }.reduce(0, +) / total
        
    let e: Float = 0.00005 //this is a standard smoothing term
    let standardDeviation = sqrt(movingVariance + e)
      
    standardDeviations.append(standardDeviation)
  
    let normalized = activations.map { ($0 - movingMean) / standardDeviation }
    
    normalizedActivations = normalized
    
    let normalizedScaledAndShifted = normalized.map { gamma * $0 + beta }
    
    return normalizedScaledAndShifted
  }
  
  func backward(gradient: [Float]) -> [Float] {
    let dBeta = gradient.reduce(0, +)
    
    guard gradient.count == normalizedActivations.count else {
      return gradient
      //fatalError("no idea but this is broken?") //tesitng only
    }
    
    //this isnt getting called =(
    print("GOT TO BACKWARD")
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
    
    for gIndex in 0..<gradient.count {
      
      let std = standardDeviations[gIndex]
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
