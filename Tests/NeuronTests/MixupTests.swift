//
//  MixupTests.swift
//
//
//  Created by Claude on 2/11/26.
//

import Foundation
import XCTest
import NumSwift
@testable import Neuron

final class MixupTests: XCTestCase {
  
  // MARK: - Initialization Tests
  
  func testDefaultInitialization() {
    let mixup = Mixup()
    // Default alpha should be 0.2, verify by checking augmentation works
    let input: TensorBatch = [Tensor([[[1.0]]]), Tensor([[[2.0]]])]
    let labels: TensorBatch = [Tensor([[[1.0, 0.0]]]), Tensor([[[0.0, 1.0]]])]
    
    let result = mixup.augment(input, labels: labels)
    
    XCTAssertEqual(result.mixed.count, input.count)
    XCTAssertEqual(result.mixedLabels.count, labels.count)
    XCTAssertTrue(result.lambda >= 0 && result.lambda <= 1)
  }
  
  func testCustomAlphaInitialization() {
    let mixup = Mixup(alpha: 0.5)
    let input: TensorBatch = [Tensor([[[1.0]]]), Tensor([[[2.0]]])]
    let labels: TensorBatch = [Tensor([[[1.0, 0.0]]]), Tensor([[[0.0, 1.0]]])]
    
    let result = mixup.augment(input, labels: labels)
    
    XCTAssertTrue(result.lambda >= 0 && result.lambda <= 1)
  }
  
  func testZeroAlpha() {
    let mixup = Mixup(alpha: 0)
    let input: TensorBatch = [Tensor([[[1.0]]]), Tensor([[[2.0]]])]
    let labels: TensorBatch = [Tensor([[[1.0, 0.0]]]), Tensor([[[0.0, 1.0]]])]
    
    let result = mixup.augment(input, labels: labels)
    
    // When alpha is 0, lambda should be 1.0
    XCTAssertEqual(result.lambda, 1.0)
  }
  
  // MARK: - Augmentation Tests
  
  func testAugmentOutputCount() {
    let mixup = Mixup(alpha: 0.2)
    let batchSize = 8
    
    let input: TensorBatch = (0..<batchSize).map { _ in
      Tensor.fillRandom(size: TensorSize(rows: 4, columns: 4, depth: 3))
    }
    let labels: TensorBatch = (0..<batchSize).map { _ in
      Tensor([[[1.0, 0.0, 0.0]]])
    }
    
    let result = mixup.augment(input, labels: labels)
    
    XCTAssertEqual(result.mixed.count, batchSize)
    XCTAssertEqual(result.mixedLabels.count, batchSize)
  }
  
  func testAugmentPreservesShape() {
    let mixup = Mixup(alpha: 0.2)
    let inputSize = TensorSize(rows: 8, columns: 8, depth: 3)
    
    let input: TensorBatch = [
      Tensor.fillRandom(size: inputSize),
      Tensor.fillRandom(size: inputSize),
      Tensor.fillRandom(size: inputSize)
    ]
    let labels: TensorBatch = [
      Tensor([[[1.0, 0.0]]]),
      Tensor([[[0.0, 1.0]]]),
      Tensor([[[1.0, 0.0]]])
    ]
    
    let result = mixup.augment(input, labels: labels)
    
    for mixed in result.mixed {
      XCTAssertEqual(TensorSize(array: mixed.shape), inputSize)
    }
  }
  
  func testLambdaInValidRange() {
    let mixup = Mixup(alpha: 0.4)
    
    // Run multiple times to test randomness
    for _ in 0..<10 {
      let input: TensorBatch = [Tensor([[[1.0]]]), Tensor([[[2.0]]])]
      let labels: TensorBatch = [Tensor([[[1.0]]]), Tensor([[[0.0]]])]
      
      let result = mixup.augment(input, labels: labels)
      
      XCTAssertGreaterThanOrEqual(result.lambda, 0)
      XCTAssertLessThanOrEqual(result.lambda, 1)
    }
  }
  
  // MARK: - Mixup Formula Tests
  
  func testMixupFormulaWithKnownLambda() {
    // Test adjustForAugment which uses the formula: lambda * a + (1 - lambda) * b
    let mixup = Mixup(alpha: 0)  // alpha=0 gives lambda=1
    
    let a = Tensor([[[2.0, 4.0]]])
    let b = Tensor([[[6.0, 8.0]]])
    
    // First call augment to set lambda
    let input: TensorBatch = [a, b]
    let labels: TensorBatch = [Tensor([[[1.0]]]), Tensor([[[0.0]]])]
    let result = mixup.augment(input, labels: labels)
    
    // With lambda=1: result = 1*a + 0*b = a
    let adjusted = mixup.adjustForAugment(a, b)
    
    XCTAssertEqual(result.lambda, 1.0)
    XCTAssertEqual(adjusted.storage[0], 2.0, accuracy: 0.0001)
    XCTAssertEqual(adjusted.storage[1], 4.0, accuracy: 0.0001)
  }
  
  func testAdjustForAugmentConsistency() {
    let mixup = Mixup(alpha: 0.2)
    
    let a = Tensor([[[1.0, 2.0, 3.0]]])
    let b = Tensor([[[4.0, 5.0, 6.0]]])
    
    // Call augment to set lambda
    let input: TensorBatch = [a, b]
    let labels: TensorBatch = [Tensor([[[1.0]]]), Tensor([[[0.0]]])]
    let result = mixup.augment(input, labels: labels)
    
    let adjusted = mixup.adjustForAugment(a, b)
    let lambda = result.lambda
    
    // Verify the formula: lambda * a + (1 - lambda) * b
    let expected0 = lambda * 1.0 + (1 - lambda) * 4.0
    let expected1 = lambda * 2.0 + (1 - lambda) * 5.0
    let expected2 = lambda * 3.0 + (1 - lambda) * 6.0
    
    XCTAssertEqual(adjusted.storage[0], expected0, accuracy: 0.0001)
    XCTAssertEqual(adjusted.storage[1], expected1, accuracy: 0.0001)
    XCTAssertEqual(adjusted.storage[2], expected2, accuracy: 0.0001)
  }
  
  // MARK: - Augmenter Enum Tests
  
  func testAugmenterEnumMixup() {
    let augmenter = Augmenter.mixup(0.3)
    let augmenting = augmenter.augmenting
    
    XCTAssertTrue(augmenting is Mixup)
    
    let input: TensorBatch = [Tensor([[[1.0]]]), Tensor([[[2.0]]])]
    let labels: TensorBatch = [Tensor([[[1.0]]]), Tensor([[[0.0]]])]
    
    let result = augmenting.augment(input, labels: labels)
    
    XCTAssertEqual(result.mixed.count, 2)
  }
  
  // MARK: - Edge Cases
  
  func testSingleItemBatch() {
    let mixup = Mixup(alpha: 0.2)
    
    let input: TensorBatch = [Tensor([[[1.0, 2.0, 3.0]]])]
    let labels: TensorBatch = [Tensor([[[1.0, 0.0]]])]
    
    let result = mixup.augment(input, labels: labels)
    
    XCTAssertEqual(result.mixed.count, 1)
    XCTAssertEqual(result.mixedLabels.count, 1)
  }
  
  func testLargeBatch() {
    let mixup = Mixup(alpha: 0.2)
    let batchSize = 64
    let inputSize = TensorSize(rows: 16, columns: 16, depth: 3)
    
    let input: TensorBatch = (0..<batchSize).map { _ in
      Tensor.fillRandom(size: inputSize)
    }
    let labels: TensorBatch = (0..<batchSize).map { i in
      var labelData = [Tensor.Scalar](repeating: 0, count: 10)
      labelData[i % 10] = 1.0
      return Tensor([[[Tensor.Scalar](labelData)]])
    }
    
    let result = mixup.augment(input, labels: labels)
    
    XCTAssertEqual(result.mixed.count, batchSize)
    XCTAssertEqual(result.mixedLabels.count, batchSize)
  }
  
  // MARK: - AugmentedDatasetModel Tests
  
  func testAugmentedDatasetModelContainsExpectedData() {
    let mixup = Mixup(alpha: 0.2)
    
    let input: TensorBatch = [
      Tensor([[[1.0, 2.0]]]),
      Tensor([[[3.0, 4.0]]]),
      Tensor([[[5.0, 6.0]]])
    ]
    let labels: TensorBatch = [
      Tensor([[[1.0, 0.0, 0.0]]]),
      Tensor([[[0.0, 1.0, 0.0]]]),
      Tensor([[[0.0, 0.0, 1.0]]])
    ]
    
    let result = mixup.augment(input, labels: labels)
    
    // Verify structure
    XCTAssertFalse(result.mixed.isEmpty)
    XCTAssertFalse(result.mixedLabels.isEmpty)
    XCTAssertTrue(result.lambda >= 0 && result.lambda <= 1)
    
    // Mixed data should have same structure as input
    for (original, mixed) in zip(input, result.mixed) {
      XCTAssertEqual(TensorSize(array: original.shape), TensorSize(array: mixed.shape))
    }
  }
  
  // MARK: - Protocol Conformance Tests
  
  func testAugmentingProtocolConformance() {
    let mixup: Augmenting = Mixup(alpha: 0.2)
    
    let input: TensorBatch = [Tensor([[[1.0]]]), Tensor([[[2.0]]])]
    let labels: TensorBatch = [Tensor([[[1.0]]]), Tensor([[[0.0]]])]
    
    // Test augment method
    let result = mixup.augment(input, labels: labels)
    XCTAssertEqual(result.mixed.count, input.count)
    
    // Test adjustForAugment method
    let adjusted = mixup.adjustForAugment(Tensor([[[1.0]]]), Tensor([[[2.0]]]))
    XCTAssertEqual(adjusted.shape, [1, 1, 1])
  }
}
