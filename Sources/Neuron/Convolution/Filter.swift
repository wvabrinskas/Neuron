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
  internal var deltas: [[Float]] = []
  
  private var optimizer: OptimizerFunction?
  private var learningRate: Float
  private var filterSize: TensorSize
  private var bias: Float
  
  internal init(size: TensorSize,
                inputSize: TensorSize,
                optimizer: OptimizerFunction? = nil,
                initializer: Initializer,
                learningRate: Float,
                bias: Float = 1.0) {
   
    self.bias = bias
    
    for _ in 0..<size.2 {
      var kernel: [[Float]] = []
      
      for _ in 0..<size.0 {
        var filterRow: [Float] = []
        
        for _ in 0..<size.1 {
          let weight = initializer.calculate(input: inputSize.depth * size.rows * size.columns,
                                             out: inputSize.depth * size.rows * size.columns)
          filterRow.append(weight)
        }
        
        kernel.append(filterRow)
      }
      
      kernels.append(kernel)
    }
    
    self.filterSize = size
    self.learningRate = learningRate
    self.optimizer = optimizer
  }
  
  internal func flip180() -> [[[Float]]] {
    kernels.map { $0.flip180() }
  }
  
  //input depth should equal the filter depth
  //return 1D array to apply activations more easily
  internal func apply(to inputs: [[[Float]]], inputSize: TensorSize) -> [Float] {
    guard inputs.count == kernels.count else {
      fatalError("filter depth does not match input depth")
      //return []
    }
        
    var convolved: [Float] = []
    for i in 0..<inputs.count {
      let currentFilter = kernels[i]
      let input = inputs[i]
      
      let conv = input.conv2D(currentFilter,
                              filterSize: (filterSize.rows, filterSize.columns),
                              inputSize: (inputSize.rows, inputSize.columns))
      
      if convolved.isEmpty {
        convolved = conv
      } else {
        convolved = convolved + conv
      }
    }
    
    //save some calculations if bias is == 0
    if bias != .zero {
      return convolved + bias
    }
    
    return convolved
  }
  
  internal func setDeltas(deltas: [[Float]]) {
    self.deltas = deltas
  }
  
  internal func setGradients(gradients: [[[Float]]]) {
    self.gradients = gradients
  }
  
  internal func zeroGradients() {
    deltas = []
    gradients = []
  }
  
  internal func clear() {
    deltas = []
    gradients = []
  }
  
  internal func adjustWeights(batchSize: Int) {
    
    let summedDeltas = deltas.flatMap { $0 }.sum 
    bias -= learningRate * summedDeltas
    
    for f in 0..<gradients.count {
      let currentFilter = kernels[f]
      let filterGradient = gradients[f]

      var newKernel: [[Float]] = []
      for row in 0..<filterGradient.count {
        let currentFilterRow = currentFilter[row]
        let currentGradientRow = filterGradient[row] / Float(batchSize)
        
        var newFilterRow: [Float] = []

        if let optimizer = optimizer {
          
          for i in 0..<currentFilterRow.count {
            let g = currentGradientRow[i]
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
