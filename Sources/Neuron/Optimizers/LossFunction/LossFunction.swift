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
  /// Mean squared error: average of squared differences between prediction and target.
  ///
  /// Formula: `sum((predicted - correct)^2) / N`
  case meanSquareError
  /// Cross-entropy loss used without a preceding Softmax layer.
  ///
  /// Use when the network's final layer does NOT include `Softmax`. Applies `−log(p)` directly.
  case crossEntropy
  /// Cross-entropy loss optimized for use with a preceding Softmax layer.
  ///
  /// Use when the network's final layer IS a `Softmax`. The derivative simplifies to `predicted − correct`.
  case crossEntropySoftmax
  /// Cross-entropy with label smoothing, parameterized by a smoothing coefficient.
  ///
  /// Smooths the one-hot targets by mixing them with a uniform prior, reducing overconfidence.
  /// The associated value is the smoothing factor `ε ∈ (0, 1)`.
  case crossEntropySoftmaxSmoothing(Tensor.Scalar)
  /// Sparse categorical cross-entropy used without a preceding Softmax layer.
  ///
  /// Behaves like `.crossEntropy` but expects the label to be a tensor of **integer class
  /// indices** rather than a one-hot encoded distribution. This avoids materializing large
  /// one-hot label tensors when the number of classes is high.
  ///
  /// The label tensor is expected to carry one class index per prediction distribution. For a
  /// prediction of size `(columns: C, rows: R, depth: D)` the label should be of size
  /// `(columns: 1, rows: R, depth: D)`, where each value is the target class index in `0..<C`.
  case sparseCrossEntropy
  /// Sparse categorical cross-entropy optimized for use with a preceding Softmax layer.
  ///
  /// Behaves like `.crossEntropySoftmax` (its derivative simplifies to `predicted − oneHot(index)`)
  /// but expects the label to be a tensor of **integer class indices** rather than a one-hot
  /// encoded distribution. See `.sparseCrossEntropy` for the expected label layout.
  case sparseCrossEntropySoftmax
  /// Binary cross-entropy loss used without a preceding Softmax layer.
  ///
  /// Suitable for binary or multi-label classification tasks. Does not assume a preceding activation.
  case binaryCrossEntropy
  /// Binary cross-entropy loss optimized for use with a preceding Softmax layer.
  ///
  /// Use when the network's final layer IS a `Softmax`. Shares the simplified derivative of `.crossEntropySoftmax`.
  case binaryCrossEntropySoftmax
  /// Wasserstein distance loss used for WGAN training.
  ///
  /// The critic returns the raw dot product of predictions and labels (no log). Use `.minimaxBinaryCrossEntropy` for standard GANs.
  case wasserstein
  /// Minimax binary cross-entropy used for standard GAN discriminator training.
  ///
  /// Equivalent to binary cross-entropy; typically paired with a sigmoid output layer.
  case minimaxBinaryCrossEntropy
  /// Focal loss with softmax, down-weighting easy examples using `alpha` and `gamma`.
  ///
  /// Reduces the relative loss for well-classified examples, focusing training on hard cases.
  /// - `alpha`: Class-balancing weight factor.
  /// - `gamma`: Focusing exponent; higher values suppress easy examples more aggressively.
  case focalSoftmax(alpha: Tensor.Scalar, gamma: Tensor.Scalar)
  /// Huber (smooth L1) loss with a transition threshold `delta`.
  ///
  /// Behaves like MSE for small errors (`|e| ≤ delta`) and like L1 for large errors,
  /// making it less sensitive to outliers than pure MSE.
  case huber(delta: Tensor.Scalar)
  
  
  /// Indicates whether this loss expects labels as integer class indices rather than
  /// one-hot encoded distributions.
  ///
  /// Consumers that interpret labels themselves (accuracy calculation, for instance) must
  /// branch on this: a sparse label slice holds the class index as its *value*, so taking
  /// its `indexOfMax` always yields `0`.
  public var isSparse: Bool {
    switch self {
    case .sparseCrossEntropy, .sparseCrossEntropySoftmax:
      return true
    default:
      return false
    }
  }

  /// Calculate the loss given a prediction tensor and a label tensor.
  /// - Parameters:
  ///   - predicted: Tensor to compare against the label
  ///   - correct: The label wrt to the predicted tensor. Expects the number of classes to be defined in the column count of the tensor.
  ///   eg. [[[1, 0, 0]]] = 3 classes
  ///   eg. [[[1], [1], [1]]] = 1 class
  ///   eg. [[[1, 0], [0, 1]]] = 2 classes
  /// - Returns: The loss as a tensor object.
  ///   - ignoring: Optional class index whose positions are excluded from the loss entirely --
  ///     typically the padding token, so a padded batch is not scored on predicting padding.
  ///     Sparse variants only.
  public func calculate(_ predicted: Tensor, correct: Tensor, ignoring ignoreIndex: Int? = nil) -> Tensor {
    // Sparse variants accept a label tensor of integer class indices whose shape intentionally
    // differs from the prediction, so they are handled before the one-hot shape check.
    switch self {
    case .sparseCrossEntropy, .sparseCrossEntropySoftmax:
      return calculateSparse(predicted, correct: correct, ignoring: ignoreIndex)
    default:
      break
    }

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

  /// Calculates the sparse categorical cross-entropy loss.
  ///
  /// - Parameters:
  ///   - predicted: Predicted probability distributions of size `(columns: C, rows: R, depth: D)`.
  ///   - correct: Integer class indices of size `(columns: 1, rows: R, depth: D)`; each value
  ///     is the target class index in `0..<C` for the corresponding distribution.
  /// - Returns: A tensor of per-distribution losses of size `(columns: R, rows: 1, depth: D)`,
  ///   mirroring the layout produced by the one-hot cross-entropy path.
  private func calculateSparse(_ predicted: Tensor, correct: Tensor, ignoring ignoreIndex: Int? = nil) -> Tensor {
    let size = predicted.size
    let depth = size.depth
    let rows = size.rows
    let cols = size.columns

    let expectedLabelCount = depth * rows
    guard correct.storage.count >= expectedLabelCount else {
      fatalError("sparse cross entropy expects \(expectedLabelCount) class indices, received \(correct.storage.count)")
    }

    let resultStorage = TensorStorage.create(count: depth * rows)

    // Average over the positions that actually count. With nothing ignored this is exactly the
    // previous divisor (`depth`), since `validCount` is then `depth * rows`.
    var validCount = 0
    if let ignoreIndex {
      for i in 0..<expectedLabelCount where Int(correct.storage[i]) != ignoreIndex {
        validCount += 1
      }
    } else {
      validCount = expectedLabelCount
    }

    guard validCount > 0 else {
      return Tensor(storage: resultStorage, size: TensorSize(rows: 1, columns: rows, depth: depth))
    }

    let divisor = Tensor.Scalar(validCount) / Tensor.Scalar(max(1, rows))

    for d in 0..<depth {
      let depthOffset = d * rows * cols
      let labelDepthOffset = d * rows
      for r in 0..<rows {
        let classIndex = Int(correct.storage[labelDepthOffset + r])

        // Ignored positions keep their zero: they contribute no loss at all.
        if let ignoreIndex, classIndex == ignoreIndex { continue }

        guard classIndex >= 0 && classIndex < cols else {
          fatalError("sparse cross entropy class index \(classIndex) is out of range 0..<\(cols)")
        }

        let p = predicted.storage[depthOffset + r * cols + classIndex]
        let loss = (-1 * Tensor.Scalar.log(p + .stabilityFactor)) / divisor
        resultStorage[d * rows + r] = loss
      }
    }

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
  ///   - ignoring: Optional class index whose positions produce a zero gradient row, so
  ///     padded timesteps contribute nothing to the update. Sparse variants only.
  public func derivative(_ predicted: Tensor, correct: Tensor, ignoring ignoreIndex: Int? = nil) -> Tensor {
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

    case .sparseCrossEntropySoftmax:
      // Equivalent to the one-hot softmax derivative `predicted - correct`, but the one-hot
      // vector is reconstructed on the fly from the sparse class indices.
      let size = predicted.size
      let cols = size.columns
      let rows = size.rows
      let depth = size.depth

      let result = predicted.storage.copy()
      for d in 0..<depth {
        let depthOffset = d * rows * cols
        let labelDepthOffset = d * rows
        for r in 0..<rows {
          let classIndex = Int(correct.storage[labelDepthOffset + r])

          // An ignored position gets a zero row rather than `predicted - oneHot`. Leaving
          // `predicted` in place would push every logit down at that timestep.
          if let ignoreIndex, classIndex == ignoreIndex {
            let rowOffset = depthOffset + r * cols
            for c in 0..<cols { result[rowOffset + c] = 0 }
            continue
          }

          guard classIndex >= 0 && classIndex < cols else {
            fatalError("sparse cross entropy class index \(classIndex) is out of range 0..<\(cols)")
          }
          let idx = depthOffset + r * cols + classIndex
          result[idx] = result[idx] - 1
        }
      }
      return Tensor(storage: result, size: size)

    case .sparseCrossEntropy:
      // Matches `.crossEntropy`: `-1 / p` at the true class index, zero elsewhere.
      let size = predicted.size
      let cols = size.columns
      let rows = size.rows
      let depth = size.depth

      let result = TensorStorage.create(count: cols * rows * depth)
      for d in 0..<depth {
        let depthOffset = d * rows * cols
        let labelDepthOffset = d * rows
        for r in 0..<rows {
          let classIndex = Int(correct.storage[labelDepthOffset + r])

          // Storage starts zeroed, so skipping an ignored position leaves its row at zero.
          if let ignoreIndex, classIndex == ignoreIndex { continue }

          guard classIndex >= 0 && classIndex < cols else {
            fatalError("sparse cross entropy class index \(classIndex) is out of range 0..<\(cols)")
          }
          let idx = depthOffset + r * cols + classIndex
          result[idx] = -1 / (predicted.storage[idx] + .stabilityFactor)
        }
      }
      return Tensor(storage: result, size: size)
      
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
    // Sparse variants receive a single class index in `correct`, so their element count
    // intentionally differs from `predicted`. Handle them before the equal-count check.
    switch self {
    case .sparseCrossEntropy, .sparseCrossEntropySoftmax:
      guard let first = correct.first else { return 0 }
      let classIndex = Int(first)
      guard classIndex >= 0 && classIndex < predicted.count else { return 0 }
      return -1 * Tensor.Scalar.log(predicted[classIndex] + .stabilityFactor)
    default:
      break
    }

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
      
    case .sparseCrossEntropy, .sparseCrossEntropySoftmax:
      // Handled above, before the equal-count guard, because the sparse label carries a single
      // class index rather than a full one-hot vector.
      return 0

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
