import XCTest
@testable import Neuron

final class TensorStorageTests: XCTestCase {

  // MARK: - Initialization

  func testInitCount() {
    let storage = TensorStorage(count: 5)
    XCTAssertEqual(storage.count, 5)
    for i in 0..<5 {
      XCTAssertEqual(storage[i], 0)
    }
  }

  func testInitEmpty() {
    let storage = TensorStorage(count: 0)
    XCTAssertEqual(storage.count, 0)
    XCTAssertTrue(storage.isEmpty)
  }

  func testInitRepeating() {
    let storage = TensorStorage(repeating: 3.14, count: 4)
    XCTAssertEqual(storage.count, 4)
    for i in 0..<4 {
      XCTAssertEqual(storage[i], 3.14, accuracy: 0.001)
    }
  }

  func testInitFromArray() {
    let source: [Tensor.Scalar] = [1, 2, 3, 4, 5]
    let storage = TensorStorage(source)
    XCTAssertEqual(storage.count, 5)
    for i in 0..<5 {
      XCTAssertEqual(storage[i], source[i])
    }
  }

  func testInitFromEmptyArray() {
    let storage = TensorStorage([Tensor.Scalar]())
    XCTAssertEqual(storage.count, 0)
    XCTAssertTrue(storage.isEmpty)
  }

  func testInitFromContiguousArray() {
    let source: ContiguousArray<Tensor.Scalar> = [10, 20, 30]
    let storage = TensorStorage(source)
    XCTAssertEqual(storage.count, 3)
    XCTAssertEqual(storage[0], 10)
    XCTAssertEqual(storage[1], 20)
    XCTAssertEqual(storage[2], 30)
  }

  // MARK: - Subscript

  func testSubscriptGetSet() {
    let storage = TensorStorage(count: 3)
    storage[0] = 1.0
    storage[1] = 2.0
    storage[2] = 3.0
    XCTAssertEqual(storage[0], 1.0)
    XCTAssertEqual(storage[1], 2.0)
    XCTAssertEqual(storage[2], 3.0)
  }

  // MARK: - Bridging

  func testToContiguousArray() {
    let source: [Tensor.Scalar] = [1, 2, 3, 4]
    let storage = TensorStorage(source)
    let result = storage.toContiguousArray()
    XCTAssertEqual(Array(result), source)
  }

  func testToArray() {
    let source: [Tensor.Scalar] = [5, 6, 7]
    let storage = TensorStorage(source)
    XCTAssertEqual(storage.toArray(), source)
  }

  func testEmptyBridging() {
    let storage = TensorStorage(count: 0)
    XCTAssertEqual(storage.toArray(), [])
    XCTAssertEqual(Array(storage.toContiguousArray()), [])
  }

  // MARK: - Copy

  func testCopyIsIndependent() {
    let original = TensorStorage([1, 2, 3])
    let copied = original.copy()

    XCTAssertEqual(original, copied)

    copied[0] = 99
    XCTAssertEqual(original[0], 1)
    XCTAssertEqual(copied[0], 99)
    XCTAssertNotEqual(original, copied)
  }

  func testCopyEmpty() {
    let original = TensorStorage(count: 0)
    let copied = original.copy()
    XCTAssertEqual(original, copied)
    XCTAssertTrue(copied.isEmpty)
  }

  // MARK: - Equatable

  func testEqualSameData() {
    let a = TensorStorage([1, 2, 3])
    let b = TensorStorage([1, 2, 3])
    XCTAssertEqual(a, b)
  }

  func testNotEqualDifferentData() {
    let a = TensorStorage([1, 2, 3])
    let b = TensorStorage([1, 2, 4])
    XCTAssertNotEqual(a, b)
  }

  func testNotEqualDifferentCount() {
    let a = TensorStorage([1, 2])
    let b = TensorStorage([1, 2, 3])
    XCTAssertNotEqual(a, b)
  }

  func testEqualBothEmpty() {
    let a = TensorStorage(count: 0)
    let b = TensorStorage(count: 0)
    XCTAssertEqual(a, b)
  }

  // MARK: - Collection Conformance

  func testSequenceIteration() {
    let storage = TensorStorage([10, 20, 30])
    var collected: [Tensor.Scalar] = []
    for value in storage {
      collected.append(value)
    }
    XCTAssertEqual(collected, [10, 20, 30])
  }

  func testCollectionIndices() {
    let storage = TensorStorage([1, 2, 3, 4])
    XCTAssertEqual(storage.startIndex, 0)
    XCTAssertEqual(storage.endIndex, 4)
    XCTAssertEqual(storage.count, 4)
  }

  func testCollectionMap() {
    let storage = TensorStorage([1, 2, 3])
    let doubled = storage.map { $0 * 2 }
    XCTAssertEqual(doubled, [2, 4, 6])
  }

  func testCollectionReduce() {
    let storage = TensorStorage([1, 2, 3, 4])
    let sum = storage.reduce(Tensor.Scalar(0), +)
    XCTAssertEqual(sum, 10)
  }

  func testCollectionEnumerated() {
    let storage = TensorStorage([10, 20, 30])
    for (i, val) in storage.enumerated() {
      XCTAssertEqual(val, storage[i])
    }
  }

  func testRandomAccessSlice() {
    let storage = TensorStorage([0, 1, 2, 3, 4, 5])
    let slice = storage[2..<5]
    XCTAssertEqual(Array(slice), [2, 3, 4])
  }

  func testMutableCollectionAssignment() {
    let storage = TensorStorage([1, 2, 3])
    storage[1] = 99
    XCTAssertEqual(storage.toArray(), [1, 99, 3])
  }

  // MARK: - Unsafe Access

  func testWithUnsafeBufferPointer() {
    let storage = TensorStorage([1, 2, 3])
    let sum = storage.withUnsafeBufferPointer { buf -> Tensor.Scalar in
      buf.reduce(0, +)
    }
    XCTAssertEqual(sum, 6)
  }

  func testWithUnsafeMutableBufferPointer() {
    let storage = TensorStorage([1, 2, 3])
    storage.withUnsafeMutableBufferPointer { buf in
      for i in 0..<buf.count {
        buf[i] *= 10
      }
    }
    XCTAssertEqual(storage.toArray(), [10, 20, 30])
  }

  // MARK: - Codable

  func testCodableRoundTrip() throws {
    let original = TensorStorage([1.5, 2.5, 3.5, 4.5])
    let encoder = JSONEncoder()
    let data = try encoder.encode(original)

    let decoder = JSONDecoder()
    let decoded = try decoder.decode(TensorStorage.self, from: data)

    XCTAssertEqual(original, decoded)
    XCTAssertEqual(decoded.count, 4)
  }

  func testCodableEmpty() throws {
    let original = TensorStorage(count: 0)
    let encoder = JSONEncoder()
    let data = try encoder.encode(original)

    let decoder = JSONDecoder()
    let decoded = try decoder.decode(TensorStorage.self, from: data)

    XCTAssertEqual(decoded.count, 0)
    XCTAssertTrue(decoded.isEmpty)
  }

  // MARK: - External Pointer Init

  func testExternalPointerInit() {
    let count = 3
    let ptr = UnsafeMutablePointer<Tensor.Scalar>.allocate(capacity: count)
    ptr.initialize(repeating: 42, count: count)

    var didDeallocate = false
    let storage = TensorStorage(pointer: ptr, count: count, deallocator: {
      ptr.deinitialize(count: count)
      ptr.deallocate()
      didDeallocate = true
    })

    XCTAssertEqual(storage[0], 42)
    XCTAssertEqual(storage[1], 42)
    XCTAssertEqual(storage[2], 42)
    XCTAssertEqual(storage.count, 3)

    // Verify the pointer is the same (not a copy)
    XCTAssertEqual(storage.pointer, ptr)

    // Force deallocation
    withExtendedLifetime(storage) {}
  }

  // MARK: - Debug Description

  func testDebugDescription() {
    let storage = TensorStorage([1, 2, 3])
    let desc = storage.debugDescription
    XCTAssertTrue(desc.contains("TensorStorage[3]"))
  }

  func testDebugDescriptionLong() {
    let storage = TensorStorage(repeating: 1, count: 20)
    let desc = storage.debugDescription
    XCTAssertTrue(desc.contains("20 total"))
  }
  
  func testBatch() {
    let tensorsToAdd = 2
    let batchSize = tensorsToAdd + 1 // we start with 1

    var tensor = Tensor.fillRandom(size: .init(array: [3,3,1]))
    var batchTensors: TensorBatch = [tensor]
    
    for _ in 0..<tensorsToAdd {
      let newTensor = Tensor.fillRandom(size: .init(array: [3,3,1]))
      batchTensors.append(newTensor)
      tensor = tensor.concat(newTensor, axis: 3)
    }
    
    XCTAssertEqual(tensor.size.batchCount, batchSize)
    
    for i in 0..<batchSize {
      let batchedTensor = tensor.batchSlice(i)
      let expectedTensor = batchTensors[i]
      XCTAssertTrue(expectedTensor.isValueEqual(to: batchedTensor, accuracy: 0.00001))
    }
    
  }
}
