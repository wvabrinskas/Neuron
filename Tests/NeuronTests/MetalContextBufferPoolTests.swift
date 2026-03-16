import XCTest
@testable import Neuron
import Metal

final class MetalContextBufferPoolTests: XCTestCase {

  func testMetalContextSingleton() {
    let a = MetalContext.shared
    let b = MetalContext.shared
    XCTAssertTrue(a === b)
  }

  func testMetalContextIsAvailable() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    XCTAssertTrue(MetalContext.shared.isAvailable)
    XCTAssertNotNil(MetalContext.shared.device)
    XCTAssertNotNil(MetalContext.shared.commandQueue)
  }

  func testBufferPoolAcquireReleaseReuse() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    let device = MetalContext.shared.device!
    let pool = BufferPool(device: device)

    let b1 = pool.acquire(byteCount: 256)
    XCTAssertNotNil(b1)
    XCTAssertGreaterThanOrEqual(b1!.length, 256)

    pool.release(b1!)

    let b2 = pool.acquire(byteCount: 256)
    XCTAssertNotNil(b2)
    XCTAssertTrue(b1 === b2, "Should reuse released buffer")
  }

  func testBufferPoolBucketing() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    let device = MetalContext.shared.device!
    let pool = BufferPool(device: device)

    let b1 = pool.acquire(byteCount: 100)
    let b2 = pool.acquire(byteCount: 100)
    XCTAssertNotNil(b1)
    XCTAssertNotNil(b2)
    XCTAssertGreaterThanOrEqual(b1!.length, 100)
    XCTAssertGreaterThanOrEqual(b2!.length, 100)

    pool.release(b1!)
    pool.release(b2!)

    let b3 = pool.acquire(byteCount: 100)
    let b4 = pool.acquire(byteCount: 100)
    XCTAssertNotNil(b3)
    XCTAssertNotNil(b4)
    XCTAssertTrue(b3 === b1 || b3 === b2)
    XCTAssertTrue(b4 === b1 || b4 === b2)
    XCTAssertTrue(b3 !== b4)
  }

  func testBufferPoolDrain() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    let device = MetalContext.shared.device!
    let pool = BufferPool(device: device)

    let b1 = pool.acquire(byteCount: 64)
    pool.release(b1!)

    pool.drain()

    let b2 = pool.acquire(byteCount: 64)
    XCTAssertNotNil(b2)
    XCTAssertTrue(b1 !== b2, "After drain, should allocate new buffer")
  }

  func testBufferPoolPowerOfTwoSizing() throws {
    try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal not available")
    let device = MetalContext.shared.device!
    let pool = BufferPool(device: device)

    let b = pool.acquire(byteCount: 500)
    XCTAssertNotNil(b)
    XCTAssertEqual(b!.length, 512, "Should round up to next power of 2")
  }
}
