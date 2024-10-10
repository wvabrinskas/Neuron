//
//  ModelTests.swift
//  Neuron
//
//  Created by William Vabrinskas on 9/14/24.
//

import XCTest
import NumSwift
@testable import Neuron

final class ModelUtilitiesTests: XCTestCase {
  
  func test_movingMean_zeros() {
    let movingMean = MovingMean(count: 10)
    XCTAssertEqual(movingMean.mean, 0)
  }
  
  func test_movingMean_append() {
    let movingMean = MovingMean(count: 5)
    movingMean.append(1)
    XCTAssertEqual(movingMean.mean, 0.2)
    movingMean.append(2)
    XCTAssertEqual(movingMean.mean, 0.6)
  }
  
  func test_movingMean_append_dropsFirst() {
    let movingMean = MovingMean(count: 5, value: [1,1,1,1,1])
    XCTAssertEqual(movingMean.mean, 1)
    movingMean.append(2)
    XCTAssertEqual(movingMean.mean, 1.2)
    XCTAssertEqual(movingMean.value, [1,1,1,1,2])
  }
}
