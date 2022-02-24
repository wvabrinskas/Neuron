//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/22/22.
//

import Foundation
import NumSwift

internal class Filter {
  internal var kernels: [[[Float]]] = []
  internal var gradients: [[[Float]]] = []
  
  private var optimizer: OptimizerFunction?
  private var learningRate: Float
  
  internal init(size: (Int, Int, Int),
                optimizer: OptimizerFunction? = nil,
                learningRate: Float) {
    let distribution = NormalDistribution(mean: 0, deviation: 0.1)

    for _ in 0..<size.2 {
      var kernel: [[Float]] = []
      
      for _ in 0..<size.0 {
        var filterRow: [Float] = []
        
        for _ in 0..<size.1 {
          let weight = distribution.nextFloat()
          filterRow.append(weight)
        }
        
        kernel.append(filterRow)
      }
      
      kernels.append(kernel)
    }
    
    self.learningRate = learningRate
    self.optimizer = optimizer
  }
  
  internal func flip180() -> [[[Float]]] {
    kernels.map { $0.flip180() }
  }
  
  //input depth should equal the filter depth
  //return 1D array to apply activations more easily
  internal func apply(to inputs: [[[Float]]]) -> [Float] {
    guard inputs.count == kernels.count else {
      fatalError("filter depth does not match input depth")
      //return []
    }
        
    var convolved: [Float] = []
    for i in 0..<inputs.count {
      let currentFilter = kernels[i]
      let input = inputs[i]
      
      let conv = input.conv2D(currentFilter)
      
      if convolved.isEmpty {
        convolved = conv
      } else {
        convolved = convolved + conv
      }
    }
    
    return convolved
  }
  
  internal func setGradients(gradients: [[[Float]]]) {
    self.gradients = gradients
  }
  
  internal func zeroGradients() {
    gradients.removeAll()
  }
  
  internal func clear() {
    gradients.removeAll()
  }
  
  internal func adjustWeights(batchSize: Int) {
    
    for f in 0..<gradients.count {
      let currentFilter = kernels[f]
      let filterGradient = gradients[f]

      var newKernel: [[Float]] = []
      for row in 0..<filterGradient.count {
        let currentFilterRow = currentFilter[row]
        let currentGradientRow = filterGradient[row]
        
        var newFilterRow: [Float] = []

        if let optimizer = optimizer {
          
          for i in 0..<currentFilterRow.count {
            let g = currentGradientRow[i] / Float(batchSize)
            let w = currentFilterRow[i]
            let newWeight = optimizer.run(weight: w, gradient: g)
            newFilterRow.append(newWeight)
          }
          
        } else {
          let adjustFilterGradient = currentGradientRow * learningRate
          newFilterRow = currentFilterRow - adjustFilterGradient
        }
        
        newKernel.append(newFilterRow)
      }
    
      kernels[f] = newKernel
    }
    
  }
}
