//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/30/22.
//

import Foundation
import NumSwift

/// An enumeration of common loss functions used for training neural networks.
/// Each case represents a different strategy for measuring prediction error.
public enum LossFunction {
  case meanSquareError
  case crossEntropy
  case crossEntropySoftmax
  case crossEntropySoftmaxSmoothing(Tensor.Scalar)
  case binaryCrossEntropy
  case binaryCrossEntropySoftmax
  case wasserstein
  case minimaxBinaryCrossEntropy
  case focalSoftmax(alpha: Tensor.Scalar, gamma: Tensor.Scalar)
  case huber(delta: Tensor.Scalar)
  
  
  /// Calculate the loss given a prediction tensor and a label tensor.
  /// - Parameters:
  ///   - predicted: Tensor to compare against the label
  ///   - correct: The label wrt to the predicted tensor. Expects the number of classes to be defined in the column count of the tensor.
  ///   eg. [[[1, 0, 0]]] = 3 classes
  ///   eg. [[[1], [1], [1]]] = 1 class
  ///   eg. [[[1, 0], [0, 1]]] = 2 classes
  /// - Returns: The loss as a tensor object.
  public func calculate(_ predicted: Tensor, correct: Tensor) -> Tensor {
    guard predicted.shape == correct.shape else {
      fatalError("predicted shape does not match correct shape")
    }
    
    let size = predicted.size
    let depth = size.depth
    let rows = size.rows
    let cols = size.columns
        
    // Build result using flat storage
    let resultStorage = TensorStorage.create(count: depth * 1 * rows)
    let depthScalar = Tensor.Scalar(depth)
    
    for d in 0..<depth {
      let depthOffset = d * rows * cols
      for r in 0..<rows {
        let rowOffset = depthOffset + r * cols
        
        // Extract row slices for predicted and correct
        var p = [Tensor.Scalar](repeating: 0, count: cols)
        var c = [Tensor.Scalar](repeating: 0, count: cols)
        for col in 0..<cols {
          p[col] = predicted.storage[rowOffset + col]
          c[col] = correct.storage[rowOffset + col]
        }
        
        let loss = calculate(p, correct: c) / depthScalar
        // Result shape: [depth][1][rows] → flat index: d * 1 * rows + 0 * rows + r
        resultStorage[d * rows + r] = loss
      }
    }
    
    // Output shape: each depth has 1 row with `rows` columns (matching how rows map to loss values)
    let resultSize = TensorSize(rows: 1, columns: rows, depth: depth)
    return Tensor(storage: resultStorage, size: resultSize)
  }
  
  /// Calculate the derivative of the loss given a prediction tensor and a label tensor.
  /// - Parameters:
  ///   - predicted: Tensor to compare against the label
  ///   - correct: The label wrt to the predicted tensor. Expects the number of classes to be defined in the column count of the tensor.
  ///   eg. [[[1, 0, 0]]] = 3 classes
  ///   eg. [[[1], [1], [1]]] = 1 class
  ///   eg. [[[1, 0], [0, 1]]] = 2 classes
  /// - Returns: The loss as a tensor object.
  public func derivative(_ predicted: Tensor, correct: Tensor) -> Tensor {
    switch self {
    case .huber(let delta):
      let size = predicted.size
      let cols = size.columns
      let rows = size.rows
      let depth = size.depth
      let result = TensorStorage.create(count: cols * rows * depth)
      let C = Tensor.Scalar(cols * rows * depth)
      
      var i = 0
      zip(predicted.storage, correct.storage).forEach { (pred, true_) in
        let e = true_ - pred          // residual
        let absE = abs(e)
        
        let grad: Tensor.Scalar = if absE <= delta {
          -e                 // quadratic region: -residual
        } else {
          -delta * (e > 0 ? 1.0 : -1.0)   // linear region: clamped
        }
        
        let val = grad / C              // account for the mean reduction
        result[i] = val
        i += 1
      }
      return Tensor(storage: result, size: predicted.size)
      
    case .focalSoftmax(let alpha, let gamma):
      let size = predicted.size
      let cols = size.columns
      let rows = size.rows
      let depth = size.depth
      let result = TensorStorage.create(count: cols * rows * depth)
      
      for d in 0..<depth {
        let depthOffset = d * rows * cols
        for r in 0..<rows {
          let rowOffset = depthOffset + r * cols
          
          var pt: Tensor.Scalar = .stabilityFactor
          for c in 0..<cols where correct.storage[rowOffset + c] > 0 {
            pt = max(min(predicted.storage[rowOffset + c], 1 - .stabilityFactor), .stabilityFactor)
            break
          }
          
          let oneMinusPt = max(1 - pt, Tensor.Scalar.stabilityFactor)
          // G = α · (1-p_t)^(γ-1) · [(1-p_t) - γ·p_t·log(p_t)]
          let G = alpha * Tensor.Scalar.pow(oneMinusPt, gamma - 1)
                * (oneMinusPt - gamma * pt * Tensor.Scalar.log(pt))
          
          for c in 0..<cols {
            // (p - y), not (y - p)
            result[rowOffset + c] = G * (predicted.storage[rowOffset + c] - correct.storage[rowOffset + c])
          }
        }
      }
      
      return Tensor(storage: result, size: size)

    case .meanSquareError:
      let N = Tensor.Scalar(predicted.size.columns * predicted.size.rows * predicted.size.depth)
      return 2 * (predicted - correct) / N
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
    case .crossEntropySoftmaxSmoothing(let smoothing):
      //only if Softmax is the modifier
      let totalClasses = Tensor.Scalar(predicted.size.columns)

      let smoothedCorreect = (1 - smoothing) * correct + smoothing / totalClasses
      return predicted - smoothedCorreect
    }
  }
  /// Calculates scalar loss for one prediction vector and label vector.
  ///
  /// - Parameters:
  ///   - predicted: Predicted probabilities/logits for one sample.
  ///   - correct: Ground-truth target values for the same sample.
  /// - Returns: Scalar loss value.
  public func calculate(_ predicted: [Tensor.Scalar], correct: [Tensor.Scalar]) -> Tensor.Scalar {
    guard predicted.count == correct.count else {
      return 0
    }
    
    switch self {
    case .huber(let delta):
      let elementWise = zip(predicted, correct).map { (pred, true_) -> Tensor.Scalar in
        let e = true_ - pred
        let absE = abs(e)
        return absE <= delta ? 0.5 * e * e : delta * (absE - 0.5 * delta)
      }
      return elementWise.reduce(0, +) / Tensor.Scalar(elementWise.count)
      
    case .focalSoftmax(let alpha, let gamma):
      // we just need index of max because we multiply by the label and in one-hot labels
      // all other's result in 0
      // we take the value of predicted that corresponds to the one hot label value.
      let indexOfMax = Int(correct.indexOfMax.0)
      let p = predicted[indexOfMax]
      let pClamped = max(min(p, 1 - .stabilityFactor), .stabilityFactor)

      return -alpha * Tensor.Scalar.pow((1 - pClamped), gamma) * Tensor.Scalar.log(pClamped)
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
      // we just need index of max because we multiply by the label and in one-hot labels
      // all other's result in 0
      // we take the value of predicted that corresponds to the one hot label value.
      let indexOfMax = Int(correct.indexOfMax.0)
      let p = predicted[indexOfMax]

      return -1 * Tensor.Scalar.log(p + .stabilityFactor)
            
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
    case .crossEntropySoftmaxSmoothing(let smoothing):
      var sum: Tensor.Scalar = 0
      let totalClasses = Tensor.Scalar(predicted.count)

      for i in 0..<predicted.count {
        let predictedScalar = predicted[i]
        let correct = (1 - smoothing) * correct[i] + smoothing / totalClasses
        sum += -1 * (correct * Tensor.Scalar.log(predictedScalar + .stabilityFactor))
      }
      
      return sum
    }

  }

}
