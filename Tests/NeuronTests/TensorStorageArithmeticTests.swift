import XCTest
@testable import Neuron
import NumSwift

final class TensorStorageArithmeticTests: XCTestCase {

  // MARK: - Element-wise (TensorStorage x TensorStorage)

  func testAdd() {
    let a = TensorStorage([1, 2, 3, 4, 5] as [Tensor.Scalar])
    let b = TensorStorage([10, 20, 30, 40, 50] as [Tensor.Scalar])
    let result = a + b
    XCTAssertEqual(result.toArray(), [11, 22, 33, 44, 55])
  }

  func testSub() {
    let a = TensorStorage([10, 20, 30, 40, 50] as [Tensor.Scalar])
    let b = TensorStorage([1, 2, 3, 4, 5] as [Tensor.Scalar])
    let result = a - b
    XCTAssertEqual(result.toArray(), [9, 18, 27, 36, 45])
  }

  func testMul() {
    let a = TensorStorage([1, 2, 3, 4, 5] as [Tensor.Scalar])
    let b = TensorStorage([2, 3, 4, 5, 6] as [Tensor.Scalar])
    let result = a * b
    XCTAssertEqual(result.toArray(), [2, 6, 12, 20, 30])
  }

  func testDiv() {
    let a = TensorStorage([10, 20, 30, 40, 50] as [Tensor.Scalar])
    let b = TensorStorage([2, 4, 5, 8, 10] as [Tensor.Scalar])
    let result = a / b
    XCTAssertEqual(result.toArray(), [5, 5, 6, 5, 5])
  }

  // MARK: - Matches ContiguousArray operators

  func testAddMatchesContiguousArray() {
    let aArr: ContiguousArray<Tensor.Scalar> = [1.5, 2.5, 3.5, 4.5]
    let bArr: ContiguousArray<Tensor.Scalar> = [0.5, 1.0, 1.5, 2.0]
    let expected = aArr + bArr

    let a = TensorStorage(aArr)
    let b = TensorStorage(bArr)
    let result = a + b
    XCTAssertEqual(result.toContiguousArray(), expected)
  }

  func testSubMatchesContiguousArray() {
    let aArr: ContiguousArray<Tensor.Scalar> = [10, 20, 30, 40]
    let bArr: ContiguousArray<Tensor.Scalar> = [1, 2, 3, 4]
    let expected = aArr - bArr

    let a = TensorStorage(aArr)
    let b = TensorStorage(bArr)
    let result = a - b
    XCTAssertEqual(result.toContiguousArray(), expected)
  }

  func testMulMatchesContiguousArray() {
    let aArr: ContiguousArray<Tensor.Scalar> = [1, 2, 3, 4]
    let bArr: ContiguousArray<Tensor.Scalar> = [5, 6, 7, 8]
    let expected = aArr * bArr

    let a = TensorStorage(aArr)
    let b = TensorStorage(bArr)
    let result = a * b
    XCTAssertEqual(result.toContiguousArray(), expected)
  }

  func testDivMatchesContiguousArray() {
    let aArr: ContiguousArray<Tensor.Scalar> = [10, 20, 30, 40]
    let bArr: ContiguousArray<Tensor.Scalar> = [2, 4, 5, 8]
    let expected = aArr / bArr

    let a = TensorStorage(aArr)
    let b = TensorStorage(bArr)
    let result = a / b
    XCTAssertEqual(result.toContiguousArray(), expected)
  }

  // MARK: - Scalar Arithmetic (TensorStorage x Scalar)

  func testAddScalar() {
    let a = TensorStorage([1, 2, 3] as [Tensor.Scalar])
    let result = a + Tensor.Scalar(10)
    XCTAssertEqual(result.toArray(), [11, 12, 13])
  }

  func testSubScalar() {
    let a = TensorStorage([10, 20, 30] as [Tensor.Scalar])
    let result = a - Tensor.Scalar(5)
    XCTAssertEqual(result.toArray(), [5, 15, 25])
  }

  func testMulScalar() {
    let a = TensorStorage([1, 2, 3] as [Tensor.Scalar])
    let result = a * Tensor.Scalar(3)
    XCTAssertEqual(result.toArray(), [3, 6, 9])
  }

  func testDivScalar() {
    let a = TensorStorage([10, 20, 30] as [Tensor.Scalar])
    let result = a / Tensor.Scalar(5)
    XCTAssertEqual(result.toArray(), [2, 4, 6])
  }

  // MARK: - Scalar Arithmetic (Scalar x TensorStorage)

  func testScalarMul() {
    let a = TensorStorage([1, 2, 3] as [Tensor.Scalar])
    let result = Tensor.Scalar(3) * a
    XCTAssertEqual(result.toArray(), [3, 6, 9])
  }

  func testScalarSub() {
    let a = TensorStorage([1, 2, 3] as [Tensor.Scalar])
    let result = Tensor.Scalar(10) - a
    XCTAssertEqual(result.toArray(), [9, 8, 7])
  }

  func testScalarDiv() {
    let a = TensorStorage([2, 4, 5] as [Tensor.Scalar])
    let result = Tensor.Scalar(100) / a
    XCTAssertEqual(result.toArray(), [50, 25, 20])
  }

  func testScalarAdd() {
    let a = TensorStorage([1, 2, 3] as [Tensor.Scalar])
    let result = Tensor.Scalar(10) + a
    XCTAssertEqual(result.toArray(), [11, 12, 13])
  }

  // MARK: - Scalar matches ContiguousArray

  func testScalarArithmeticMatchesContiguousArray() {
    let arr: ContiguousArray<Tensor.Scalar> = [1, 2, 3, 4]
    let scalar: Tensor.Scalar = 5
    let storage = TensorStorage(arr)

    XCTAssertEqual((storage + scalar).toContiguousArray(), arr + scalar)
    XCTAssertEqual((storage - scalar).toContiguousArray(), arr - scalar)
    XCTAssertEqual((storage * scalar).toContiguousArray(), arr * scalar)
    XCTAssertEqual((storage / scalar).toContiguousArray(), arr / scalar)
    XCTAssertEqual((scalar * storage).toContiguousArray(), scalar * arr)
    XCTAssertEqual((scalar - storage).toContiguousArray(), scalar - arr)
  }

  // MARK: - Reductions

  func testSum() {
    let a = TensorStorage([1, 2, 3, 4, 5] as [Tensor.Scalar])
    XCTAssertEqual(a.sum, 15, accuracy: 0.001)
  }

  func testMean() {
    let a = TensorStorage([1, 2, 3, 4, 5] as [Tensor.Scalar])
    XCTAssertEqual(a.mean, 3, accuracy: 0.001)
  }

  func testSumOfSquares() {
    let a = TensorStorage([1, 2, 3] as [Tensor.Scalar])
    XCTAssertEqual(a.sumOfSquares, 14, accuracy: 0.001)
  }

  func testReductionsMatchContiguousArray() {
    let arr: ContiguousArray<Tensor.Scalar> = [1.5, 2.5, 3.5, 4.5]
    let storage = TensorStorage(arr)
    XCTAssertEqual(storage.sum, arr.sum, accuracy: 0.001)
    XCTAssertEqual(storage.mean, arr.mean, accuracy: 0.001)
    XCTAssertEqual(storage.sumOfSquares, arr.sumOfSquares, accuracy: 0.001)
  }

  func testEmptyReductions() {
    let a = TensorStorage(count: 0)
    XCTAssertEqual(a.sum, 0)
    XCTAssertEqual(a.mean, 0)
    XCTAssertEqual(a.sumOfSquares, 0)
  }

  // MARK: - Unary Operations

  func testNegated() {
    let a = TensorStorage([1, -2, 3, -4] as [Tensor.Scalar])
    let result = a.negated()
    XCTAssertEqual(result.toArray(), [-1, 2, -3, 4])
  }

  func testSquareRoot() {
    let a = TensorStorage([1, 4, 9, 16, 25] as [Tensor.Scalar])
    let result = a.squareRoot()
    let expected: [Tensor.Scalar] = [1, 2, 3, 4, 5]
    for i in 0..<expected.count {
      XCTAssertEqual(result[i], expected[i], accuracy: 0.001)
    }
  }

  func testClipped() {
    let a = TensorStorage([-5, -1, 0, 1, 5] as [Tensor.Scalar])
    let result = a.clipped(to: 2)
    XCTAssertEqual(result.toArray(), [-2, -1, 0, 1, 2])
  }

  func testEmptyUnary() {
    let a = TensorStorage(count: 0)
    XCTAssertTrue(a.negated().isEmpty)
    XCTAssertTrue(a.squareRoot().isEmpty)
    XCTAssertTrue(a.clipped(to: 1).isEmpty)
  }

  // MARK: - Independence

  func testArithmeticDoesNotMutateOperands() {
    let a = TensorStorage([1, 2, 3] as [Tensor.Scalar])
    let b = TensorStorage([4, 5, 6] as [Tensor.Scalar])
    let _ = a + b
    XCTAssertEqual(a.toArray(), [1, 2, 3])
    XCTAssertEqual(b.toArray(), [4, 5, 6])
  }
}
