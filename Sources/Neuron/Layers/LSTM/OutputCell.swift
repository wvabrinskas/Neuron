//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/23/23.
//

import Foundation
import NumSwift


class OutputCell {
  let device: Device
  
  struct Parameters {
    var hiddenOutputWeights: Tensor
    var hiddenOutputBiases: Tensor
    var activationMatrix: Tensor
  }
  
  init(device: Device = CPU()) {
    self.device = device
  }

  func forward(parameters: Parameters) -> Tensor {
    var outputMatrix = parameters.activationMatrix.matmul(parameters.hiddenOutputWeights) + parameters.hiddenOutputBiases.asScalar()
    outputMatrix = Softmax(inputSize: .init(array: outputMatrix.shape)).forward(tensor: outputMatrix)
    return outputMatrix
  }
  
  func backward(gradient: [[Tensor.Scalar]],
                activations: [[Tensor.Scalar]],
                batchSize: Int,
                hiddenOutputWeights: Tensor) -> (outputs: Tensor, weights: Tensor, biases: Tensor) {
    let w = hiddenOutputWeights.value.transpose2d()
    
    let wrtOutputs = Tensor(gradient).matmul(Tensor(w))
    
    var wrtWeights = Tensor(activations.transpose2d()).matmul(Tensor(gradient))
    
    let wrtBiases = Tensor(gradient).sum(axis: 1)
    
    if batchSize > 1 {
      wrtWeights = wrtWeights / Tensor.Scalar(batchSize)
    }
    
    return (wrtOutputs, wrtWeights, wrtBiases)
  }
}
