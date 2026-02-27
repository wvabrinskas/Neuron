//
//  InitializerTests.swift
//
//  Created by William Vabrinskas on 2/25/26.
//

import XCTest
@testable import Neuron

final class InitializerTests: XCTestCase {

  // MARK: - Orthogonal: Shape

  func test_orthogonal_outputShape_matchesRequestedSize() {
    let size = TensorSize(rows: 4, columns: 4, depth: 2)
    let initializer = InitializerType.orthogonal(gain: 1.0).build()
    let tensor = initializer.calculate(size: size, input: 4, out: 4)
    XCTAssertEqual(TensorSize(array: tensor.shape), size)
  }

  func test_orthogonal_outputShape_tallMatrix() {
    let size = TensorSize(rows: 6, columns: 3, depth: 1)
    let initializer = InitializerType.orthogonal(gain: 1.0).build()
    let tensor = initializer.calculate(size: size, input: 6, out: 3)
    XCTAssertEqual(TensorSize(array: tensor.shape), size)
  }

  func test_orthogonal_outputShape_wideMatrix() {
    let size = TensorSize(rows: 3, columns: 6, depth: 1)
    let initializer = InitializerType.orthogonal(gain: 1.0).build()
    let tensor = initializer.calculate(size: size, input: 3, out: 6)
    XCTAssertEqual(TensorSize(array: tensor.shape), size)
  }

  // MARK: - Orthogonal: Orthonormality

  /// Verifies that rows of the generated matrix are orthonormal (Q^T * Q ≈ I).
  func test_orthogonal_rows_areOrthonormal() {
    let rows = 4
    let cols = 4
    let size = TensorSize(rows: rows, columns: cols, depth: 1)
    let initializer = InitializerType.orthogonal(gain: 1.0).build()
    let tensor = initializer.calculate(size: size, input: rows, out: cols)

    // Check each pair of rows: dot product should be ~0 for distinct rows, ~1 for same row.
    for i in 0..<rows {
      for j in 0..<rows {
        let d = rowDot(tensor: tensor, row: i, otherRow: j, depth: 0, cols: cols)
        if i == j {
          XCTAssertEqual(d, 1.0, accuracy: 1e-5, "Row \(i) should be unit length")
        } else {
          XCTAssertEqual(d, 0.0, accuracy: 1e-5, "Rows \(i) and \(j) should be orthogonal")
        }
      }
    }
  }

  /// For a tall matrix (rows > cols), only min(rows,cols) rows can be orthonormal.
  func test_orthogonal_tallMatrix_firstRowsAreOrthonormal() {
    let rows = 5
    let cols = 3
    let size = TensorSize(rows: rows, columns: cols, depth: 1)
    let initializer = InitializerType.orthogonal(gain: 1.0).build()
    let tensor = initializer.calculate(size: size, input: rows, out: cols)

    let orthonormalCount = min(rows, cols)

    for i in 0..<orthonormalCount {
      for j in 0..<orthonormalCount {
        let d = rowDot(tensor: tensor, row: i, otherRow: j, depth: 0, cols: cols)
        if i == j {
          XCTAssertEqual(d, 1.0, accuracy: 1e-5, "Row \(i) should be unit length")
        } else {
          XCTAssertEqual(d, 0.0, accuracy: 1e-5, "Rows \(i) and \(j) should be orthogonal")
        }
      }
    }
  }

  // MARK: - Orthogonal: Gain scaling

  func test_orthogonal_gain_scalesRowNorms() {
    let gain: Tensor.Scalar = 2.0
    let rows = 4
    let cols = 4
    let size = TensorSize(rows: rows, columns: cols, depth: 1)
    let initializer = InitializerType.orthogonal(gain: gain).build()
    let tensor = initializer.calculate(size: size, input: rows, out: cols)

    for i in 0..<rows {
      let norm = rowMagnitude(tensor: tensor, row: i, depth: 0, cols: cols)
      XCTAssertEqual(norm, gain, accuracy: 1e-5, "Row \(i) norm should equal gain \(gain)")
    }
  }

  func test_orthogonal_gain_negativeOne_preservesOrthogonality() {
    let gain: Tensor.Scalar = -1.0
    let rows = 3
    let cols = 3
    let size = TensorSize(rows: rows, columns: cols, depth: 1)
    let initializer = InitializerType.orthogonal(gain: gain).build()
    let tensor = initializer.calculate(size: size, input: rows, out: cols)

    for i in 0..<rows {
      for j in 0..<rows {
        let d = rowDot(tensor: tensor, row: i, otherRow: j, depth: 0, cols: cols)
        if i == j {
          // ||gain * v||^2 = gain^2 * ||v||^2 = 1
          XCTAssertEqual(d, 1.0, accuracy: 1e-6)
        } else {
          XCTAssertEqual(d, 0.0, accuracy: 1e-6)
        }
      }
    }
  }

  // MARK: - Orthogonal: Multiple depth slices are independent

  func test_orthogonal_multipleDepthSlices_eachIsOrthonormal() {
    let depth = 3
    let rows = 4
    let cols = 4
    let size = TensorSize(rows: rows, columns: cols, depth: depth)
    let initializer = InitializerType.orthogonal(gain: 1.0).build()
    let tensor = initializer.calculate(size: size, input: rows, out: cols)

    for d in 0..<depth {
      for i in 0..<rows {
        for j in 0..<rows {
          let dotProduct = rowDot(tensor: tensor, row: i, otherRow: j, depth: d, cols: cols)
          if i == j {
            XCTAssertEqual(dotProduct, 1.0, accuracy: 1e-5, "Depth \(d) row \(i) should be unit length")
          } else {
            XCTAssertEqual(dotProduct, 0.0, accuracy: 1e-5, "Depth \(d) rows \(i),\(j) should be orthogonal")
          }
        }
      }
    }
  }

  // MARK: - Orthogonal: Encode / Decode

  func test_orthogonal_encode_decode_preservesGain() {
    let expectedGain: Tensor.Scalar = 1.5
    let rawInitializer: InitializerType = .orthogonal(gain: expectedGain)
    let initializer = rawInitializer.build()

    let encoder = JSONEncoder()
    let data = try? encoder.encode(initializer)
    XCTAssertNotNil(data)

    let decoder = JSONDecoder()
    let decoded = try? decoder.decode(Initializer.self, from: data!)
    XCTAssertNotNil(decoded)

    switch decoded!.type {
    case .orthogonal(let gain):
      XCTAssertEqual(gain, expectedGain, accuracy: 1e-5)
    default:
      XCTFail("Expected .orthogonal type after decode, got \(decoded!.type)")
    }
  }

  // MARK: - Helpers

  /// Extracts a row from a tensor as a flat array of scalars.
  private func row(from tensor: Tensor, row r: Int, depth d: Int, cols: Int) -> [Tensor.Scalar] {
    (0..<cols).map { c in tensor[c, r, d] }
  }

  /// Dot product between two rows of a tensor.
  private func rowDot(tensor: Tensor, row i: Int, otherRow j: Int, depth d: Int, cols: Int) -> Tensor.Scalar {
    let a = row(from: tensor, row: i, depth: d, cols: cols)
    let b = row(from: tensor, row: j, depth: d, cols: cols)
    return zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
  }

  /// Euclidean norm of a row in a tensor.
  private func rowMagnitude(tensor: Tensor, row r: Int, depth d: Int, cols: Int) -> Tensor.Scalar {
    let v = row(from: tensor, row: r, depth: d, cols: cols)
    return Tensor.Scalar.sqrt(v.reduce(0) { $0 + $1 * $1 })
  }
}
