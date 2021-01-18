//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/18/21.
//

import Foundation
import GameplayKit

struct NormalDistribution {
  private let randomSource: GKRandomSource
  let mean: Float
  let deviation: Float
  
  init(randomSource: GKRandomSource = GKRandomSource(), mean: Float = 0, deviation: Float = 1) {
    precondition(deviation >= 0)
    self.randomSource = randomSource
    self.mean = mean
    self.deviation = deviation
    
  }
  
  func nextFloat() -> Float {
    guard deviation > 0 else { return mean }
    
    let x1 = randomSource.nextUniform()
    let x2 = randomSource.nextUniform()
    let z1 = sqrt(-2 * log(x1)) * cos(2 * Float.pi * x2)
    
    return z1 * deviation + mean
  }
}
