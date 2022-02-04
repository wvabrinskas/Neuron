//
//  LossFunction.swift
//  Nameley
//
//  Created by William Vabrinskas on 12/26/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//


import Foundation

public enum LossFunction {
  case meanSquareError
  case crossEntropy
  case binaryCrossEntropy
  
  public func calculate(_ predicted: [Float], correct: [Float]) -> Float {
    guard predicted.count == correct.count else {
      return 0
    }
    
    switch self {
    case .meanSquareError:
      var sum: Float = 0
      
      for i in 0..<predicted.count {
        let predicted = predicted[i]
        let correct = correct[i]
        let sq = pow(predicted - correct, 2)
        sum += sq
      }
      
      return sum / Float(predicted.count)
      
    case .crossEntropy:
      var sum: Float = 0

      for i in 0..<predicted.count {
        let predicted = predicted[i]
        let correct = correct[i]
        
        let p = correct == 0 ? 1 - predicted : predicted
        sum += (correct * log2(p + 1e-10))
      }
      
      return -sum
      
    case .binaryCrossEntropy:
      guard correct.count == 1 else {
        return 0
      }
      
      let y = correct[0]
      let p = predicted[0]

      func clipped(_ value: Float) -> Float {
        return max(1e-10, value)
      }
      
      let result = -1 * (y * log(clipped(p)) + (1 - y) * log(clipped(1 - p)))
      return result
    }

  }
  
  public func derivative(_ predicted: Float, correct: Float) -> Float {
    switch self {
    case .meanSquareError:
      return predicted - correct
    case .crossEntropy:
      //only if Softmax is the modifier
      //TODO: Use actual cross entropy derivate and calculate using the chain rule -> Softmax' * CrossEntropy'
      return predicted - correct
      
    case .binaryCrossEntropy:
      let y = correct
      let p = predicted
      
      let result = -1 * ((y / p) - ((1 - y) / (1 - p)))
      return result
    }
  }
}
