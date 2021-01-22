//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/21/21.
//

import Foundation
import XCTest
@testable import Neuron

public protocol BaseTestConfig: XCTestCase {
  var brain: Brain? { get set }
  func flattenedWeights() -> [Float]
}

public extension BaseTestConfig {
  
  func flattenedWeights() -> [Float] {
    guard let brain = self.brain else {
      return []
    }
    
    var flattenedWeights: [Float] = []
    
    for layer in brain.layerWeights {
      layer.forEach { (float) in
        flattenedWeights.append(float)
      }
    }
    return flattenedWeights
  }

}
