//
//  OptimizerTests.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/5/26.
//

@testable import Neuron
import XCTest


final class OptimizerTests: XCTestCase {
  
  func test_setDevice_propagatesDevice() {
    let network = Sequential(
      Dense(10, inputs: 10),
      ReLu(),
      BatchNormalize(),
      Dense(10),
      InstanceNormalize()
    )
    
    let optimizer = Adam(network, learningRate: 0.001, batchSize: 32)
    
    XCTAssert(optimizer.device.type == DeviceManager.shared.device.type)
    
    network.layers.forEach { layer in
      XCTAssert(layer.device.type == optimizer.device.type)
    }
        
    optimizer.deviceType = .gpu
    
    network.layers.forEach { layer in
      XCTAssert(layer.device.type == optimizer.device.type)
    }
  }
}
