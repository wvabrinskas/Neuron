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
  let softmax: Softmax
  
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

    dense = Dense(parameters.vocabSize, biasEnabled: true)
    dense.weights = parameters.hiddenOutputWeights
    dense.biases = parameters.hiddenOutputBiases
    dense.inputSize = TensorSize(array: [parameters.hiddenSize, 1, 1])

    softmax = Softmax(inputSize: dense.outputSize)
  }

  func forward(parameters: Parameters) -> Tensor {
    dense.weights = parameters.hiddenOutputWeights
    dense.biases = parameters.hiddenOutputBiases
    var outputMatrix = dense.forward(tensor: parameters.activationMatrix)
    outputMatrix = softmax.forward(tensor: outputMatrix)
    return outputMatrix
  }
}
