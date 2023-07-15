//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/23/23.
//

import Foundation
import NumSwift

class LSTMCell {
  let hidden: Int
  let input: Int
  let vocabSize: Int
  let device: Device
  
  struct Parameters {
    var forgetGateWeights: Tensor
    var inputGateWeights: Tensor
    var gateGateWeights: Tensor
    var outputGateWeights: Tensor
  }
  
  struct ParameterDerivatives {
    var dForgetGateWeights: Tensor
    var dInputGateWeights: Tensor
    var dGateGateWeights: Tensor
    var dOutputGateWeights: Tensor
    
    var isEmpty: Bool {
      dForgetGateWeights.isEmpty &&
      dInputGateWeights.isEmpty &&
      dGateGateWeights.isEmpty &&
      dOutputGateWeights.isEmpty
    }
    
    init(dForgetGateWeights: Tensor = .init(),
         dInputGateWeights: Tensor = .init(),
         dGateGateWeights: Tensor = .init(),
         dOutputGateWeights: Tensor = .init()) {
      self.dForgetGateWeights = dForgetGateWeights
      self.dInputGateWeights = dInputGateWeights
      self.dGateGateWeights = dGateGateWeights
      self.dOutputGateWeights = dOutputGateWeights
    }
    
    static func +(lhs: ParameterDerivatives, rhs: ParameterDerivatives) -> ParameterDerivatives {
      let newDFG = lhs.dForgetGateWeights + rhs.dForgetGateWeights
      let newDIG = lhs.dInputGateWeights + rhs.dInputGateWeights
      let newDGG = lhs.dGateGateWeights + rhs.dGateGateWeights
      let newDOG = lhs.dOutputGateWeights + rhs.dOutputGateWeights
      return .init(dForgetGateWeights: newDFG,
                   dInputGateWeights: newDIG,
                   dGateGateWeights: newDGG,
                   dOutputGateWeights: newDOG)
    }
    
    func concat() -> Tensor {
      dForgetGateWeights.concat(dInputGateWeights, axis: 2)
        .concat(dGateGateWeights, axis: 2)
        .concat(dOutputGateWeights, axis: 2)
    }
  }
  
  struct Activations {
    var fa: Tensor
    var ia: Tensor
    var oa: Tensor
    var ga: Tensor
    var activationMatrix: Tensor
    var cellMemoryMatrix: Tensor
  }
  
  struct Errors {
    struct LSTMError {
      var ef: Tensor // error forget
      var ei: Tensor // error input
      var eo: Tensor // error output
      var eg: Tensor // error gate
    }
    
    var previousActivationError: Tensor
    var previousCellError: Tensor
    var embeddingError: Tensor
  }
  
  init(hidden: Int,
       input: Int,
       vocabSize: Int,
       device: Device = CPU()) {
    self.hidden = hidden
    self.input = input
    self.device = device
    self.vocabSize = vocabSize
  }
  
  func forward(tensor: Tensor,
               parameters: Parameters,
               cache: LSTM.Cache) -> Activations {
    
    let fgw = parameters.forgetGateWeights
    let igw = parameters.inputGateWeights
    let ogw = parameters.outputGateWeights
    let ggw = parameters.gateGateWeights
    
    let previousActivationMatrix = cache.activation
    let previousCellMatrix = cache.cell
    
    let concat = tensor.concat(previousActivationMatrix)
    
    // forget gate
    let fa = device.matmul(concat, fgw)
    let faOut = Sigmoid().forward(tensor: fa)
    
    // input gate
    let ia = device.matmul(concat, igw)
    let iaOut = Sigmoid().forward(tensor: ia)
    
    // gate gate
    let ga = device.matmul(concat, ggw)
    let gaOut = Tanh().forward(tensor: ga)
    
    // output gate
    let oa = device.matmul(concat, ogw)
    let oaOut = Sigmoid().forward(tensor: oa)
    
    let cellMemoryMatrix = (faOut * previousCellMatrix) + (iaOut * gaOut)
    
    let tanOut = Tanh().forward(tensor: cellMemoryMatrix)
    
    let activationMatrix = oaOut * tanOut
    
    return Activations(fa: faOut,
                       ia: iaOut,
                       oa: oaOut,
                       ga: gaOut,
                       activationMatrix: activationMatrix,
                       cellMemoryMatrix: cellMemoryMatrix)
    
  }
  
  func backward(cache: LSTM.Cache,
                previousCache: LSTM.Cache?,
                activationOutputError: [[Tensor.Scalar]],
                nextActivationError: [[Tensor.Scalar]],
                nextCellError: [[Tensor.Scalar]],
                batchSize: Int,
                parameters: Parameters) -> (inputs: Errors, weights: ParameterDerivatives) {

    let activationError = activationOutputError + nextActivationError

    let lstm = cache.lstm
    let cellActivation = cache.cell
    let previousCellActivaion = previousCache?.cell ?? cache.cell.zerosLike() // if there's no previous state use 0s. Might need to come up with a better solution so we dont have to re-do this
    
    let tanActivationOfCellActivation = Tanh().forward(tensor: cellActivation)
    
    // output gate error
    let oa = lstm.outputGate
    var eo = Tensor(activationError) * tanActivationOfCellActivation
    eo = (eo * oa) * (1 - oa)
    
    // cell activation error
    var cellError = Tensor(activationError) * oa
    cellError = cellError * self.device.derivate(tanActivationOfCellActivation, .tanh)
    cellError = cellError + Tensor(nextCellError)
    
    // input gate error
    let ia = lstm.inputGate
    let ga = lstm.gateGate
    var ei = cellError * ia
    ei = (ei * ia) * (1 - ia)
    
    // gate gate error
    var eg = cellError * ia
    eg = eg * self.device.derivate(ga, .tanh)
    
    // forget gate error
    let fa = lstm.forgetGate
    var ef = cellError * previousCellActivaion
    ef = (ef * fa) * (1 - fa)
    
    // prev cell error
    let prevCellError = cellError * fa
    
    let fgw = parameters.forgetGateWeights
    let igw = parameters.inputGateWeights
    let ogw = parameters.outputGateWeights
    let ggw = parameters.gateGateWeights
    
    var embedActivationError = ef.matmul(Tensor(fgw.value.transpose()))
    embedActivationError = embedActivationError + (ei.matmul(Tensor(igw.value.transpose())))
    embedActivationError = embedActivationError + (eo.matmul(Tensor(ogw.value.transpose())))
    embedActivationError = embedActivationError + (eg.matmul(Tensor(ggw.value.transpose())))
    
    let fgwShape = fgw.shape
    let inputHiddenUnits = fgwShape[safe: 1] ?? 0
    let hiddenUnits = fgwShape[safe: 0] ?? 0
    let inputUnits = inputHiddenUnits - hiddenUnits
    
    let prevActivationError = embedActivationError[inputUnits..., 0..., 0...]
    
    let embedError = embedActivationError[0..<inputUnits, 0... , 0...]
    
    let errors =  Errors.LSTMError(ef: ef,
                                   ei: ei,
                                   eo: eo,
                                   eg: eg)
    
    // get derivatives wrt to weights
    let weightDerivatives = backwardsWRTWeights(lstmError: errors,
                                                embedding: cache.embedding,
                                                activation: cache.activation,
                                                batchSize: batchSize)
    
    return (inputs: Errors(previousActivationError: prevActivationError,
                           previousCellError: prevCellError,
                           embeddingError: embedError),
            weights: weightDerivatives)
  }
  
  private func backwardsWRTWeights(lstmError: Errors.LSTMError,
                                   embedding: Tensor,
                                   activation: Tensor,
                                   batchSize: Int) -> ParameterDerivatives {
    
    let transposed = activation.concat(embedding, axis: -1).value.transpose()
    
    let concat = Tensor(transposed)
    
    let ef = lstmError.ef
    let ei = lstmError.ei
    let eo = lstmError.eo
    let eg = lstmError.eg
    
    let dfgw = concat.matmul(ef) / Tensor.Scalar(batchSize)
    let digw = concat.matmul(ei) / Tensor.Scalar(batchSize)
    let dogw = concat.matmul(eo) / Tensor.Scalar(batchSize)
    let dggw = concat.matmul(eg) / Tensor.Scalar(batchSize)

    return .init(dForgetGateWeights: dfgw,
                 dInputGateWeights: digw,
                 dGateGateWeights: dggw,
                 dOutputGateWeights: dogw)
  }
}
