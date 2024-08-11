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
  let dense: Dense
  
  struct Parameters {
    var hiddenOutputWeights: Tensor
    var hiddenOutputBiases: Tensor
    var activationMatrix: Tensor
    let vocabSize: Int
    let hiddenSize: Int
  }
  
  init(device: Device = CPU(),
       parameters: Parameters) {
    self.device = device
    
    dense = Dense(parameters.vocabSize)
    dense.weights = parameters.hiddenOutputWeights
    dense.biases = parameters.hiddenOutputBiases
    dense.inputSize = TensorSize(array: [parameters.hiddenSize, 1, 1])
  }

  func forward(parameters: Parameters) -> Tensor {
    dense.weights = parameters.hiddenOutputWeights
    dense.biases = parameters.hiddenOutputBiases
    var outputMatrix = dense.forward(tensor: parameters.activationMatrix)
    outputMatrix = Softmax(inputSize: .init(array: outputMatrix.shape)).forward(tensor: outputMatrix)
    return outputMatrix
  }
}
