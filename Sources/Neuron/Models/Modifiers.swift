//
//  File.swift
//  
//
//  Created by William Vabrinskas on 12/27/20.
//

import Foundation


public enum OutputModifier: Int, CaseIterable {
  case softmax
  
  func calculate(index: Int, outputs: [Float]) -> Float {
    switch self {
    case .softmax:
      let max = outputs.max() ?? 1
      var sum: Float = 0
      outputs.forEach { (output) in
        sum += pow(Float(Darwin.M_E), output - max)
      }
      
      return pow(Float(Darwin.M_E), outputs[index] - max) / sum
    }
  }

}
