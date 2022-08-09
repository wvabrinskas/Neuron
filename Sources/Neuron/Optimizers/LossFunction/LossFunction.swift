//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/30/22.
//

import Foundation
import NumSwift

public enum LossFunction {
  case meanSquareError
  case crossEntropy
  case crossEntropySoftmax
  case binaryCrossEntropy
  case wasserstein
  case minimaxBinaryCrossEntropy
  
  public func calculate(_ predicted: [Tensor.Scalar], correct: [Tensor.Scalar]) -> Tensor.Scalar {
    guard predicted.count == correct.count else {
      return 0
    }
    
    switch self {
    case .wasserstein:
      guard correct.count == 1 && predicted.count == 1 else {
        return 0
      }
      
      return predicted[safe: 0, 0] * correct[safe: 0, 0]
      
    case .meanSquareError:
      var sum: Tensor.Scalar = 0
      
      for i in 0..<predicted.count {
        let predicted = predicted[i]
        let correct = correct[i]
        let sq = pow(predicted - correct, 2)
        sum += sq
      }
      
      return sum / Tensor.Scalar(predicted.count)
      
    case .crossEntropySoftmax, .crossEntropy:
      var sum: Tensor.Scalar = 0

      for i in 0..<predicted.count {
        let predicted = predicted[i]
        let correct = correct[i]
        sum += -1 * (correct * log(predicted + 1e-10))
      }
      
      return sum
      
    case .binaryCrossEntropy,
         .minimaxBinaryCrossEntropy:
      guard correct.count == 1 else {
        return 0
      }
      
      let y = correct[0]
      let p = predicted[0]

      func clipped(_ value: Tensor.Scalar) -> Tensor.Scalar {
        return max(1e-10, value)
      }
      
      let result = -1 * (y * log(clipped(p)) + (1 - y) * log(clipped(1 - p)))
      return result
    }

  }
  
  public func derivative(_ predicted: [Tensor.Scalar], correct: [Tensor.Scalar]) -> [Tensor.Scalar] {
    precondition(predicted.count == correct.count)
    
    switch self {
    case .meanSquareError:
      return 2 * (predicted - correct)
    case .crossEntropy:
      return predicted.map { -1 * (1 / $0) }
      
    case .crossEntropySoftmax:
      //only if Softmax is the modifier
      return predicted - correct
      
    case .binaryCrossEntropy,
         .minimaxBinaryCrossEntropy:
      let y = correct
      let p = predicted
      
      let result = -1 * ((y / p) - ((1 - y) / (1 - p)))
      return result
      
    case .wasserstein:
      return correct
    }
    
  }
}