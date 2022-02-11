//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/11/22.
//

import Foundation
import NumSwift


internal struct GradientPenalty {
  
  static func calculate(gan: GAN, real: [TrainingData], fake: [TrainingData]) -> Float {
    guard let dis = gan.discriminator else {
      return 0
    }
    
    defer {
      dis.zeroGradients()
    }
        
    var gradients: [[Float]] = []
    
    for i in 0..<real.count {
      dis.zeroGradients()
      
      let epsilon = Float.random(in: 0...1)
      var inter: [Float] = []
      if i < real.count && i < fake.count {
        let realNew = real[i].data
        let fakeNew = fake[i].data
        
        guard realNew.count == fakeNew.count else {
          return 0
        }
        
        inter = (realNew * epsilon) + (fakeNew * (1 - epsilon))
      }
      
      let output = gan.discriminate(inter)
      let loss = gan.lossFunction.loss(.real, value: output)

      dis.backpropagate(with: [loss])
      
      //skip first layer gradients
      if let networkGradients = dis.gradients()[safe: 1]?.flatMap({ $0 }) {
        gradients.append(networkGradients)
      }
    }
    
    let gradientNorm = gradients.map { $0.sumOfSquares }

    let center: Float = 1
    
    let penalty = gradientNorm.map { pow((sqrt($0) - center), 2) }.sum / (Float(gradientNorm.count) + 1e-8)
    return penalty
  }
}
