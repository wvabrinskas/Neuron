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
    var forgetGateBiases: Tensor
    var inputGateBiases: Tensor
    var gateGateBiases: Tensor
    var outputGateBiases: Tensor
  }
  
  struct ParameterDerivatives {
    var dForgetGate: Tensor
    var dInputGate: Tensor
    var dGateGate: Tensor
    var dOutputGate: Tensor

    var isEmpty: Bool {
      dForgetGate.isEmpty &&
      dInputGate.isEmpty &&
      dGateGate.isEmpty &&
      dOutputGate.isEmpty
    }
    
    init(dForgetGate: Tensor = .init(),
         dInputGate: Tensor = .init(),
         dGateGate: Tensor = .init(),
         dOutputGate: Tensor = .init()) {
      self.dForgetGate = dForgetGate
      self.dInputGate = dInputGate
      self.dGateGate = dGateGate
      self.dOutputGate = dOutputGate
    }
    
    static func +(lhs: ParameterDerivatives, rhs: ParameterDerivatives) -> ParameterDerivatives {
      let newDFG = lhs.dForgetGate + rhs.dForgetGate
      let newDIG = lhs.dInputGate + rhs.dInputGate
      let newDGG = lhs.dGateGate + rhs.dGateGate
      let newDOG = lhs.dOutputGate + rhs.dOutputGate
      return .init(dForgetGate: newDFG,
                   dInputGate: newDIG,
                   dGateGate: newDGG,
                   dOutputGate: newDOG)
    }
    
    func concat() -> Tensor {
      dForgetGate.concat(dInputGate, axis: 2)
        .concat(dGateGate, axis: 2)
        .concat(dOutputGate, axis: 2)
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
    let faOut = Sigmoid().forward(tensor: fa) + parameters.forgetGateBiases.asScalar()
    
    // input gate
    let ia = device.matmul(concat, igw)
    let iaOut = Sigmoid().forward(tensor: ia) + parameters.inputGateBiases.asScalar()
    
    // gate gate
    let ga = device.matmul(concat, ggw)
    let gaOut = Tanh().forward(tensor: ga) + parameters.gateGateBiases.asScalar()
    
    // output gate
    let oa = device.matmul(concat, ogw)
    let oaOut = Sigmoid().forward(tensor: oa) + parameters.outputGateBiases.asScalar()
    
    let cellMemoryMatrix = (faOut * previousCellMatrix) + (iaOut * gaOut)
    
    let tanOut = Tanh().forward(tensor: cellMemoryMatrix)
    
    let activationMatrix = oaOut * tanOut
    
    return Activations(fa: faOut.detached(),
                       ia: iaOut.detached(),
                       oa: oaOut.detached(),
                       ga: gaOut.detached(),
                       activationMatrix: activationMatrix.detached(),
                       cellMemoryMatrix: cellMemoryMatrix.detached())
    
  }
  
  func backward(cache: LSTM.Cache,
                previousCache: LSTM.Cache?,
                activationOutputError: [[Tensor.Scalar]],
                nextActivationError: [[Tensor.Scalar]],
                nextCellError: [[Tensor.Scalar]],
                batchSize: Int,
                parameters: Parameters) -> (inputs: Errors, weights: ParameterDerivatives, biases: ParameterDerivatives) {

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
    
    let biasDerivatives = backwarsWRTBiases(lstmError: errors,
                                            batchSize: batchSize)
    
    return (inputs: Errors(previousActivationError: prevActivationError,
                           previousCellError: prevCellError,
                           embeddingError: embedError),
            weights: weightDerivatives,
            biases: biasDerivatives)
  }
  
  private func backwarsWRTBiases(lstmError: Errors.LSTMError,
                                 batchSize: Int) -> ParameterDerivatives {
    let outputGateBiasesUpdate = lstmError.eo.sum(axis: 1)
    let inputGateBiasesUpdate = lstmError.ei.sum(axis: 1)
    let gateGateBiasesUpdate = lstmError.eg.sum(axis: 1)
    let forgetGateBiasesUpdate = lstmError.ef.sum(axis: 1)

    return .init(dForgetGate: forgetGateBiasesUpdate,
                 dInputGate: inputGateBiasesUpdate,
                 dGateGate: gateGateBiasesUpdate,
                 dOutputGate: outputGateBiasesUpdate)
  }
  
  private func backwardsWRTWeights(lstmError: Errors.LSTMError,
                                   embedding: Tensor,
                                   activation: Tensor,
                                   batchSize: Int) -> ParameterDerivatives {
    
    let transposed = embedding.concat(activation).value.transpose()
    
    let concat = Tensor(transposed)
    
    let ef = lstmError.ef
    let ei = lstmError.ei
    let eo = lstmError.eo
    let eg = lstmError.eg
    
    var dfgw = concat.matmul(ef)
    var digw = concat.matmul(ei)
    var dogw = concat.matmul(eo)
    var dggw = concat.matmul(eg)
    
    if batchSize > 1 {
      dfgw = dfgw / Tensor.Scalar(batchSize)
      digw = digw / Tensor.Scalar(batchSize)
      dogw = dogw / Tensor.Scalar(batchSize)
      dggw = dggw / Tensor.Scalar(batchSize)
    }

    return .init(dForgetGate: dfgw,
                 dInputGate: digw,
                 dGateGate: dggw,
                 dOutputGate: dogw)
  }
}
