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
  case binaryCrossEntropySoftmax
  case wasserstein
  case minimaxBinaryCrossEntropy
  
  public func calculate(_ predicted: Tensor, correct: Tensor) -> Tensor {
    guard predicted.shape == correct.shape else {
      fatalError("predicted shape does not match correct shape")
    }
    
    let depth = predicted.value.count
    var result: [[[Tensor.Scalar]]] = []
    for d in 0..<depth{
      var depthResult: [[Tensor.Scalar]] = []
      var rowResult: [Tensor.Scalar] = []
      for r in 0..<predicted.value[d].count {
        let p = predicted.value[d][r]
        let c = correct.value[d][r]
        let loss = calculate(p, correct: c) / Tensor.Scalar(depth)
        rowResult.append(loss)
      }
      depthResult.append(rowResult)
      result.append(depthResult)
    }
    
    return Tensor(result)
    
  }

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
        let sq = Tensor.Scalar.pow(predicted - correct, 2)
        sum += sq
      }
      
      return sum / Tensor.Scalar(predicted.count)
      
    case .crossEntropySoftmax, .crossEntropy:
      var sum: Tensor.Scalar = 0

      for i in 0..<predicted.count {
        let predicted = predicted[i]
        let correct = correct[i]
        sum += -1 * (correct * Tensor.Scalar.log(predicted + .stabilityFactor))
      }
      
      return sum
      
    case .binaryCrossEntropy,
         .binaryCrossEntropySoftmax,
         .minimaxBinaryCrossEntropy:
      func clipped(_ value: Tensor.Scalar) -> Tensor.Scalar {
        return max(.stabilityFactor, value)
      }
      
      var sum: Tensor.Scalar = 0
      
      for i in 0..<predicted.count {

        let y = correct[i]
        let p = predicted[i]
        sum += -1 * (y * Tensor.Scalar.log(clipped(p)) + (1 - y) * Tensor.Scalar.log(clipped(1 - p)))
      }
      
      return sum
    }

  }
  
  public func derivative(_ predicted: Tensor, correct: Tensor) -> Tensor {
    switch self {
    case .meanSquareError:
      return -1 * ((predicted - correct) * 2)
    case .crossEntropy:
      return predicted.map { -1 * (1 / $0) }
      
    case .crossEntropySoftmax,
         .binaryCrossEntropySoftmax:
      //only if Softmax is the modifier
      return predicted - correct
      
    case .binaryCrossEntropy,
         .minimaxBinaryCrossEntropy:
      let y = correct
      let p = predicted
      
      let firstDivide = y / p
      let ySubtract = Tensor.Scalar(1) - y
      let pSubtract = Tensor.Scalar(1) - p
      
      let result = -1 * ((firstDivide) - ((ySubtract) / (pSubtract)))
      return result
      
    case .wasserstein:
      return correct
    }
  }
  
  @available(*, deprecated,
              renamed: "derivative",
              message: "This will be removed soon. Please use the derivative function that accepts Tensor objects")
  public func derivative(_ predicted: [Tensor.Scalar], correct: [Tensor.Scalar]) -> [Tensor.Scalar] {
    precondition(predicted.count == correct.count)
    
    switch self {
    case .meanSquareError:
      return 2 * (predicted - correct)
    case .crossEntropy:
      return predicted.map { -1 * (1 / $0) }
      
    case .crossEntropySoftmax,
         .binaryCrossEntropySoftmax:
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
