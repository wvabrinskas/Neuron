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
  
  public func calculate(_ predicted: Float, correct: Float) -> Float {
    
    switch self {
    case .meanSquareError:
      let sq = pow(predicted - correct, 2)
      return sq
      
    case .crossEntropy:
      let p = correct == 0 ? 1 - predicted : predicted
      let result = (correct * log(p + 1e-10))
      return -result
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
