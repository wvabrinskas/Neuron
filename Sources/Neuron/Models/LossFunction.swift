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
  
  public func calculate(_ predicted: Float, correct: Float) -> Float {
    
    switch self {
    case .meanSquareError:
      let sq = pow(predicted - correct, 2)
      return sq
      
    case .crossEntropy:
      let p = correct == 0 ? 1 - predicted : predicted
      let result = (correct * log2(p + 1e-10))
      return -result
  
    case .binaryCrossEntropy:
      let y = correct
      let y1 = predicted
      
      func clipped(_ value: Float) -> Float {
        guard value < 0 else {
          return value
        }
        
        return 1e-10
      }
      
      let result = -y * log2(clipped(y1)) - (1 - y) * log2(clipped(1 - y1))
      return result
    }

  }
  
  public func derivative(_ predicted: Float, correct: Float) -> Float {
    switch self {
    case .meanSquareError:
      //−1∗(2(y−p)
      return -1 * (2 * (correct - predicted))
    case .crossEntropy:
      return predicted - correct
      
    case .binaryCrossEntropy:
      let y = correct
      let p = predicted
      
      let result = (-y / p) + ((1 - y) / (1 - p))
      return result
    }
  }
}
