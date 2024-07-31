//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/23/23.
//

import Foundation
import NumSwift


class OutputCell<N: TensorNumeric> {
  let device: Device
  
  struct Parameters {
    var hiddenOutputWeights: Tensor<N>
    var hiddenOutputBiases: Tensor<N>
    var activationMatrix: Tensor<N>
  }
  
  init(device: Device = CPU()) {
    self.device = device
  }

  func forward(parameters: Parameters) -> Tensor<N> {
    var outputMatrix = parameters.activationMatrix.matmul(parameters.hiddenOutputWeights) + parameters.hiddenOutputBiases.asScalar()
    outputMatrix = Softmax<N>(inputSize: .init(array: outputMatrix.shape)).forward(tensor: outputMatrix)
    return outputMatrix
  }
  
  func backward(gradient: [[Tensor<N>.Scalar]],
                activations: [[Tensor<N>.Scalar]],
                batchSize: Int,
                hiddenOutputWeights: Tensor<N>) -> (outputs: Tensor<N>, weights: Tensor<N>, biases: Tensor<N>) {
    let w = hiddenOutputWeights.value.transpose2d()
    
    let wrtOutputs = Tensor<N>(gradient).matmul(Tensor<N>(w))
    
    var wrtWeights = Tensor<N>(activations.transpose2d()).matmul(Tensor<N>(gradient))
    
    let wrtBiases = Tensor<N>(gradient).sum(axis: 1)
    
    if batchSize > 1 {
      wrtWeights = wrtWeights / Tensor<N>.Scalar(batchSize)
    }
    
    return (wrtOutputs, wrtWeights, wrtBiases)
  }
}
