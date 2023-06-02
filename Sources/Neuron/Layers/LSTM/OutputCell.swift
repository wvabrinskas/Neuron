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
    var activationMatrix: Tensor
  }
  
  init(device: Device = CPU()) {
    self.device = device
  }

  func forward(parameters: Parameters) -> Tensor {
    var outputMatrix = parameters.activationMatrix.matmul(parameters.hiddenOutputWeights)
    outputMatrix = Softmax(inputSize: .init(array: outputMatrix.shape)).forward(tensor: outputMatrix)
    return outputMatrix
  }
  
  func backward(gradient: [[Tensor.Scalar]],
                activations: [[Tensor.Scalar]],
                batchSize: Int,
                hiddenOutputWeights: Tensor) -> (outputs: Tensor, weights: Tensor) {
    guard let w = hiddenOutputWeights.value[safe: 0]?.transposed() else {
      fatalError("Could not transpose hidden output weights in LSTM layer")
    }
    
    let wrtOutputs = Tensor(gradient).matmul(Tensor(w))
    let wrtWeights = Tensor(activations.transposed()).matmul(Tensor(gradient)) / Tensor.Scalar(batchSize)
    return (wrtOutputs, wrtWeights)
  }
}
