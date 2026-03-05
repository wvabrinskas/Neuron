import XCTest
@testable import Neuron
import Metal

final class MetalTensorStorageTests: XCTestCase {

  var device: MTLDevice!

  override func setUp() {
    super.setUp()
    device = MTLCreateSystemDefaultDevice()
  }

  func testCreateAndRoundTrip() throws {
    try XCTSkipIf(device == nil, "Metal not available (e.g. simulator)")
    let data: [Tensor.Scalar] = [1, 2, 3, 4, 5]
    let storage = MetalTensorStorage(device: device, data: data)
    XCTAssertEqual(storage.count, 5)
    XCTAssertEqual(storage.toArray(), data)
    XCTAssertEqual(storage[0], 1)
    XCTAssertEqual(storage[4], 5)
  }

  func testMTLBufferNonNilAndLength() throws {
    try XCTSkipIf(device == nil, "Metal not available (e.g. simulator)")
    let storage = MetalTensorStorage(device: device, count: 10)
    XCTAssertNotNil(storage.mtlBuffer)
    XCTAssertEqual(storage.mtlBuffer.length, 10 * MemoryLayout<Float>.stride)
  }

  func testArithmeticMatchesCPUTensorStorage() throws {
    try XCTSkipIf(device == nil, "Metal not available (e.g. simulator)")
    let a = MetalTensorStorage(device: device, data: [1, 2, 3])
    let b = MetalTensorStorage(device: device, data: [4, 5, 6])
    let cpuA = TensorStorage([1, 2, 3])
    let cpuB = TensorStorage([4, 5, 6])

    let metalSum = a + b
    let cpuSum = cpuA + cpuB
    XCTAssertEqual(metalSum.toContiguousArray(), cpuSum.toContiguousArray())
  }

  func testPointerStability() throws {
    try XCTSkipIf(device == nil, "Metal not available (e.g. simulator)")
    let storage = MetalTensorStorage(device: device, data: [7, 8, 9])
    let ptr1 = storage.pointer
    let v = storage[1]
    let ptr2 = storage.pointer
    XCTAssertEqual(ptr1, ptr2)
    XCTAssertEqual(v, 8)
  }

  func testInitFromTensorStorage() throws {
    try XCTSkipIf(device == nil, "Metal not available (e.g. simulator)")
    let cpu = TensorStorage([10, 20, 30])
    let metal = MetalTensorStorage(device: device, storage: cpu)
    XCTAssertEqual(metal.toArray(), cpu.toArray())
  }

  func testInitCountZeroed() throws {
    try XCTSkipIf(device == nil, "Metal not available (e.g. simulator)")
    let storage = MetalTensorStorage(device: device, count: 5)
    for i in 0..<5 {
      XCTAssertEqual(storage[i], 0)
    }
  }
}
