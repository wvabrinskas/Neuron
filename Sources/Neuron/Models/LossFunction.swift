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
  case binaryCrossEntropy
  
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
      
    case .binaryCrossEntropy:
      var i = 0
      var sums: Float = 0
      
      predicted.forEach { (out) in
        let correct = correct[i]
        sums += (correct * log(out)) + (1 - correct) * log(1.0 - out)
        i += 1
      }
      
      return -(sums / Float(correct.count))
    }

  }
  
  public func derivative(_ predicted: Float, correct: Float) -> Float {
    switch self {
    case .meanSquareError:
      return correct - predicted
    case .binaryCrossEntropy:
      let yi = correct
      let yi2 = predicted
      
      let sum = -((yi / yi2) - ((1 - yi) / (1 - yi2)))
      return sum
    }
  }
}
