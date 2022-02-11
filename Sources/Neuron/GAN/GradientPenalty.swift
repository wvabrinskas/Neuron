//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/11/22.
//

import Foundation
import NumSwift


internal struct GradientPenalty {
  
  static func calculate(gradient: [Float]) -> Float {
    let sumOfSquares = gradient.sumOfSquares
    return pow(sqrt(sumOfSquares) - 1, 2)
  }
  
}
