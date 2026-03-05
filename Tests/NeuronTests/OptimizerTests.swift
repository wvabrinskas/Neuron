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
    
    XCTAssert(optimizer.device === DeviceManager.shared.device)
    
    network.layers.forEach { layer in
      XCTAssert(layer.device === optimizer.device)
    }
        
    optimizer.deviceType = .gpu
    
    network.layers.forEach { layer in
      XCTAssert(layer.device === optimizer.device)
    }
  }
}
