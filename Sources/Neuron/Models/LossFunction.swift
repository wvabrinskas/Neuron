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
  
  public func calculate(_ predicted: [Float], correct: [Float]) -> Float {
    
    switch self {
    case .meanSquareError:
      var i = 0
      var sums: Float = 0
      
      predicted.forEach { (val) in
        let correct = correct[i]
        let sq = pow(val - correct, 2)
        sums += sq
        i += 1
      }
      
      return sums / Float(correct.count)
      
    case .crossEntropy:
      var i = 0
      var sums: Float = 0
      
      predicted.forEach { (out) in
        let correctVal = correct[i]
        if predicted.count > 1 {
          sums += correctVal * log(out)
        } else {
          sums += (correctVal * log(out)) + (1 - correctVal) * log(1.0 - out)
        }
        i += 1
      }
      
      return -sums / Float(predicted.count)
    }

  }
  
  public func derivative(_ predicted: Float, correct: Float) -> Float {
    switch self {
    case .meanSquareError:
      return correct - predicted
    case .crossEntropy:
      return predicted - correct
    }
  }
}
