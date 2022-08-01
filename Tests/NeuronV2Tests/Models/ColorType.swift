//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/30/22.
//

import Foundation
public enum ColorType: Int, CaseIterable {
  case red, green, blue
  
  public var string: String {
    switch self {
    case .blue:
      return "blue"
    case .red:
      return "red"
    case .green:
      return "green"
    }
  }

  public func correctValues() -> [Float] {
    var returnArray = [Float](repeating: 0.0, count: ColorType.allCases.count)
    
    if let index = ColorType.allCases.firstIndex(of: self) {
      returnArray[index] = 1.0
    }
    
    return returnArray
  }
  
  func color() -> [Float] {
    switch self {
    case .red:
      let r = Float.random(in: 0.7...1.0)
      let g = Float.random(in: 0...0.01)
      let b = Float.random(in: 0...0.01)
      return [r, g, b, 1.0]
    case .green:
      let g = Float.random(in: 0.7...1.0)
      let r = Float.random(in: 0...0.01)
      let b = Float.random(in: 0...0.01)
      return [r, g, b, 1.0]
    case .blue:
      let b = Float.random(in: 0.7...1.0)
      let r = Float.random(in: 0...0.01)
      let g = Float.random(in: 0...0.01)
      return [r, g, b, 1.0]
    }
  }
}

public struct ColorTrainingModel: Equatable {
  var shape: ColorType
  var data: [Float]
}
