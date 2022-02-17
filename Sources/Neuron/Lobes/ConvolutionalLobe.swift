//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/16/22.
//

import Foundation
import NumSwift

public class ConvolutionalLobe: Lobe {
  
  private let poolType: PoolType
  
  public enum PoolType {
    case average, max
  }
  
  public init(model: ConvolutionalLobeModel,
              learningRate: Float) {
    
    self.poolType = model.poolingType
    super.init(model: model, learningRate: learningRate)
  }
  
  internal func pool(input: [[Float]]) -> [[Float]] {

    var rowResults: [Float] = []
    var results: [[Float]] = []
    
    let shape = input.shape
    if let rows = shape[safe: 0],
       let columns = shape[safe: 1] {
      
      for r in stride(from: 0, through: rows - 1, by: 2) {
        rowResults = []
        
        for c in stride(from: 0, through: columns - 1, by: 2) {
          let current = input[r][c]
          let right = input[r + 1][c]
          let bottom = input[r][c + 1]
          let diag = input[r + 1][c + 1]
          
          if poolType == .max {
            let max = max(max(max(current, right), bottom), diag)
            rowResults.append(max)
          } else {
            let average = (current + right + bottom + diag) / 4
            rowResults.append(average)
          }

        }
        
        results.append(rowResults)
      }
    }
    
    return results
  }
}
