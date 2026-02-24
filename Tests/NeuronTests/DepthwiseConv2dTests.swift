//
//  DepthwiseConv2dTests.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/17/26.
//

@testable import Neuron
import XCTest

class DepthwiseConv2dTests: XCTestCase {

  // MARK: - Helpers

  /// Build input tensor: shape (cols=4, rows=4, depth=2)
  /// Channel 0: values 1–16 row-major
  /// Channel 1: rows of 1,2,3,4 repeated across columns
  func makeInput() -> Tensor {
    var storage = Tensor.Value(repeating: 0, count: 4 * 4 * 2)
    // Channel 0: 1..16
    for i in 0..<16 {
      storage[i] = Tensor.Scalar(i + 1)
    }
    // Channel 1: row r+1 across all columns
    for r in 0..<4 {
      for c in 0..<4 {
        storage[16 + r * 4 + c] = Tensor.Scalar(r + 1)
      }
    }
    return Tensor(storage, size: TensorSize(rows: 4, columns: 4, depth: 2))
  }

  /// Manually set filters on the layer after initialization
  /// Filter 0: identity (center=1, rest=0)
  /// Filter 1: all-ones
  func setKnownFilters(on layer: DepthwiseConv2d) {
    let identityStorage = Tensor.Value([
      0, 0, 0,
      0, 1, 0,
      0, 0, 0
    ])
    let allOnesStorage = Tensor.Value(repeating: 1, count: 9)

    layer.filters = [
      Tensor(identityStorage, size: TensorSize(rows: 3, columns: 3, depth: 1)),
      Tensor(allOnesStorage,  size: TensorSize(rows: 3, columns: 3, depth: 1))
    ]
  }

  // MARK: - Forward Pass

  func testForwardPassShape() {
    let layer = DepthwiseConv2d(inputSize: TensorSize(rows: 4, columns: 4, depth: 2),
                                strides: (1, 1),
                                padding: .same,
                                filterSize: (3, 3),
                                biasEnabled: false)
    setKnownFilters(on: layer)

    let input = makeInput()
    let output = layer.forward(tensor: input)

    // Output shape must match input shape (same padding, stride 1)
    XCTAssertEqual(output.shape, [4, 4, 2],
                   "Output shape should be [cols=4, rows=4, depth=2]")
  }

  func testForwardPassIdentityFilter() {
    let layer = DepthwiseConv2d(inputSize: TensorSize(rows: 4, columns: 4, depth: 2),
                                strides: (1, 1),
                                padding: .same,
                                filterSize: (3, 3),
                                biasEnabled: false)
    setKnownFilters(on: layer)

    let input = makeInput()
    let output = layer.forward(tensor: input)

    // Channel 0 with identity filter: output should equal input channel 0
    let inCh0  = input.depthSlice(0)
    let outCh0 = output.depthSlice(0)
    for i in 0..<16 {
      XCTAssertEqual(outCh0[i], inCh0[i], accuracy: 1e-4,
                     "Identity filter: output[\(i)] should equal input[\(i)]")
    }
  }

  func testForwardPassAllOnesFilter() {
    let layer = DepthwiseConv2d(inputSize: TensorSize(rows: 4, columns: 4, depth: 2),
                                strides: (1, 1),
                                padding: .same,
                                filterSize: (3, 3),
                                biasEnabled: false)
    setKnownFilters(on: layer)

    let input = makeInput()
    let output = layer.forward(tensor: input)

    let outCh1 = output.depthSlice(1)  // all-ones filter channel

    // (row=0, col=0) top-left corner — only 2×2 valid window, rest zero-padded
    // valid neighbors in channel 1: [1,1, 2,2] = 6
    XCTAssertEqual(outCh1[0 * 4 + 0], 6.0, accuracy: 1e-4,
                   "Top-left corner should be 6")

    // (row=1, col=1) fully interior 3×3 window
    // channel 1 values: row0=1,row1=2,row2=3 × 3 columns = 1+1+1+2+2+2+3+3+3 = 18
    XCTAssertEqual(outCh1[1 * 4 + 1], 18.0, accuracy: 1e-4,
                   "Interior (1,1) should be 18")

    // (row=3, col=3) bottom-right corner — only 2×2 valid window
    // channel 1 values: row2=3,row3=4 × 2 cols = 3+3+4+4 = 14
    XCTAssertEqual(outCh1[3 * 4 + 3], 14.0, accuracy: 1e-4,
                   "Bottom-right corner should be 14")
  }

  func testForwardPassBiasApplied() {
    let layer = DepthwiseConv2d(inputSize: TensorSize(rows: 4, columns: 4, depth: 2),
                                strides: (1, 1),
                                padding: .same,
                                filterSize: (3, 3),
                                biasEnabled: true)
    setKnownFilters(on: layer)

    // Set biases: channel 0 = +10, channel 1 = +20
    layer.biases = Tensor(Tensor.Value([10, 20]),
                          size: TensorSize(rows: 1, columns: 2, depth: 1))

    let input = makeInput()
    let output = layer.forward(tensor: input)

    // Channel 0 identity filter + bias 10: output[0,0] = input[0,0] + 10 = 1 + 10 = 11
    XCTAssertEqual(output.depthSlice(0)[0], 11.0, accuracy: 1e-4,
                   "Identity filter + bias 10: first element should be 11")

    // Channel 1 all-ones filter + bias 20 at (0,0): 6 + 20 = 26
    XCTAssertEqual(output.depthSlice(1)[0], 26.0, accuracy: 1e-4,
                   "All-ones filter + bias 20 at corner should be 26")
  }

  // MARK: - Backward Pass

  func testBackwardPassShapes() {
    let layer = DepthwiseConv2d(inputSize: TensorSize(rows: 4, columns: 4, depth: 2),
                                strides: (1, 1),
                                padding: .same,
                                filterSize: (3, 3),
                                biasEnabled: false)
    setKnownFilters(on: layer)

    let input = makeInput()
    let upstreamGrad = Tensor(Tensor.Value(repeating: 1.0, count: 4 * 4 * 2),
                              size: TensorSize(rows: 4, columns: 4, depth: 2))

    let (dInput, dWeights, _) = layer.backward(input, upstreamGrad)

    XCTAssertEqual(dInput.shape, input.shape,
                   "dInput shape must match input shape")
    XCTAssertEqual(dWeights.shape, [3, 3, 2],
                   "dWeights shape must be [cols=3, rows=3, depth=2]")
  }

  func testBackwardPassIdentityFilterInputGrad() {
    let layer = DepthwiseConv2d(inputSize: TensorSize(rows: 4, columns: 4, depth: 2),
                                strides: (1, 1),
                                padding: .same,
                                filterSize: (3, 3),
                                biasEnabled: false)
    setKnownFilters(on: layer)

    let input = makeInput()
    let upstreamGrad = Tensor(Tensor.Value(repeating: 1.0, count: 4 * 4 * 2),
                              size: TensorSize(rows: 4, columns: 4, depth: 2))

    let (dInput, _, _) = layer.backward(input, upstreamGrad)

    // Identity filter: grad flows straight through. Every dInput channel 0
    // value should be 1.0 (upstream grad of 1 * center weight of 1)
    let dInCh0 = dInput.depthSlice(0)
    for i in 0..<16 {
      XCTAssertEqual(dInCh0[i], 1.0, accuracy: 1e-4,
                     "Identity filter dInput channel 0 [\(i)] should be 1.0")
    }
  }

  func testBackwardPassAllOnesFilterInputGrad() {
    let layer = DepthwiseConv2d(inputSize: TensorSize(rows: 4, columns: 4, depth: 2),
                                strides: (1, 1),
                                padding: .same,
                                filterSize: (3, 3),
                                biasEnabled: false)
    setKnownFilters(on: layer)

    let input = makeInput()
    let upstreamGrad = Tensor(Tensor.Value(repeating: 1.0, count: 4 * 4 * 2),
                              size: TensorSize(rows: 4, columns: 4, depth: 2))

    let (dInput, _, _) = layer.backward(input, upstreamGrad)
    let dInCh1 = dInput.depthSlice(1)

    // All-ones filter, all-ones upstream grad:
    // Each input pixel's grad = number of output pixels it influenced
    // Interior pixel (row=1, col=1) at flat index 5 → full 3×3 overlap = 9
    XCTAssertEqual(dInCh1[1 * 4 + 1], 9.0, accuracy: 1e-4,
                   "Interior pixel should receive grad 9 from all-ones filter")

    // Corner pixel (row=0, col=0) at flat index 0 → only 1 output pixel = 1
    XCTAssertEqual(dInCh1[0 * 4 + 0], 4.0, accuracy: 1e-4,
                   "Corner pixel should receive grad 1 from all-ones filter")

    // Edge pixel (row=0, col=1) → 1×2 overlap (clamped by top edge) = 2...
    // but same padding means 3 output pixels in the top row see it = 3
    XCTAssertEqual(dInCh1[0 * 4 + 1], 6.0, accuracy: 1e-4,
                   "Top-edge pixel (0,1) should receive grad 3")
  }

  func testBackwardPassFilterGrads() {
    let layer = DepthwiseConv2d(inputSize: TensorSize(rows: 4, columns: 4, depth: 2),
                                strides: (1, 1),
                                padding: .same,
                                filterSize: (3, 3),
                                biasEnabled: false)
    setKnownFilters(on: layer)

    let input = makeInput()
    let upstreamGrad = Tensor(Tensor.Value(repeating: 1.0, count: 4 * 4 * 2),
                              size: TensorSize(rows: 4, columns: 4, depth: 2))

    let (_, dWeights, _) = layer.backward(input, upstreamGrad)

    // dFilter[c, r, pos] = sum over all output positions of upstream_grad * input_patch
    // For channel 0 (input = 1..16), upstream all ones:
    // Center weight (row=1, col=1): sees all 16 input pixels → sum = 1+2+...+16 = 136
    // Flat index in dWeights channel 0: depth-slice 0, center of 3×3 = index 4
    let dWCh0 = dWeights.depthSlice(0)
    XCTAssertEqual(dWCh0[4], 136.0, accuracy: 1e-3,
                   "dFilter[0] center weight should be sum of all input ch0 = 136")

    // For channel 1 (input rows = 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4), upstream all ones:
    // Center weight: sum of all input ch1 = 4×(1+2+3+4) = 40
    let dWCh1 = dWeights.depthSlice(1)
    XCTAssertEqual(dWCh1[4], 40.0, accuracy: 1e-3,
                   "dFilter[1] center weight should be sum of all input ch1 = 40")
  }

  // MARK: - Stride 2

  func testForwardPassStride2OutputShape() {
    let layer = DepthwiseConv2d(inputSize: TensorSize(rows: 4, columns: 4, depth: 2),
                                strides: (2, 2),
                                padding: .same,
                                filterSize: (3, 3),
                                biasEnabled: false)
    setKnownFilters(on: layer)

    let input = makeInput()
    let output = layer.forward(tensor: input)

    // Same padding + stride 2 on 4×4: output should be ceil(4/2) × ceil(4/2) = 2×2
    XCTAssertEqual(output.shape, [2, 2, 2],
                   "Stride-2 same-padding output should be [2, 2, 2]")
  }
}
