//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/23/23.
//

import Foundation
import NumSwift

class LSTMCell<N: TensorNumeric> {
  let hidden: Int
  let input: Int
  let vocabSize: Int
  let device: Device
  
  struct Parameters {
    var forgetGateWeights: Tensor<N>
    var inputGateWeights: Tensor<N>
    var gateGateWeights: Tensor<N>
    var outputGateWeights: Tensor<N>
    var forgetGateBiases: Tensor<N>
    var inputGateBiases: Tensor<N>
    var gateGateBiases: Tensor<N>
    var outputGateBiases: Tensor<N>
  }
  
  struct ParameterDerivatives {
    var dForgetGate: Tensor<N>
    var dInputGate: Tensor<N>
    var dGateGate: Tensor<N>
    var dOutputGate: Tensor<N>

    var isEmpty: Bool {
      dForgetGate.isEmpty &&
      dInputGate.isEmpty &&
      dGateGate.isEmpty &&
      dOutputGate.isEmpty
    }
    
    init(dForgetGate: Tensor<N> = .init(),
         dInputGate: Tensor<N> = .init(),
         dGateGate: Tensor<N> = .init(),
         dOutputGate: Tensor<N> = .init()) {
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
    
    func concat() -> Tensor<N> {
      dForgetGate.concat(dInputGate, axis: 2)
        .concat(dGateGate, axis: 2)
        .concat(dOutputGate, axis: 2)
    }
  }
  
  struct Activations {
    var fa: Tensor<N>
    var ia: Tensor<N>
    var oa: Tensor<N>
    var ga: Tensor<N>
    var activationMatrix: Tensor<N>
    var cellMemoryMatrix: Tensor<N>
  }
  
  struct Errors {
    struct LSTMError {
      var ef: Tensor<N> // error forget
      var ei: Tensor<N> // error input
      var eo: Tensor<N> // error output
      var eg: Tensor<N> // error gate
    }
    
    var previousActivationError: Tensor<N>
    var previousCellError: Tensor<N>
    var embeddingError: Tensor<N>
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
  
  func forward(tensor: Tensor<N>,
               parameters: Parameters,
               cache: LSTM<N>.Cache) -> Activations {
    
    let fgw = parameters.forgetGateWeights
    let igw = parameters.inputGateWeights
    let ogw = parameters.outputGateWeights
    let ggw = parameters.gateGateWeights
    
    let previousActivationMatrix = cache.activation
    let previousCellMatrix = cache.cell
    
    let concat = tensor.concat(previousActivationMatrix)
    
    // forget gate
    let fa = device.matmul(concat, fgw) + parameters.forgetGateBiases.asScalar()
    let faOut = Sigmoid<N>().forward(tensor: fa)
    
    // input gate
    let ia = device.matmul(concat, igw) + parameters.inputGateBiases.asScalar()
    let iaOut = Sigmoid<N>().forward(tensor: ia)
    
    // gate gate
    let ga = device.matmul(concat, ggw) + parameters.gateGateBiases.asScalar()
    let gaOut = Tanh<N>().forward(tensor: ga)
    
    // output gate
    let oa = device.matmul(concat, ogw) + parameters.outputGateBiases.asScalar() // Could be Dense layers
    let oaOut = Sigmoid<N>().forward(tensor: oa)
    
    let cellMemoryMatrix = (faOut * previousCellMatrix) + (iaOut * gaOut)
    
    let tanOut = Tanh<N>().forward(tensor: cellMemoryMatrix)
    
    let activationMatrix = oaOut * tanOut
    
    return Activations(fa: faOut.detached(),
                       ia: iaOut.detached(),
                       oa: oaOut.detached(),
                       ga: gaOut.detached(),
                       activationMatrix: activationMatrix.detached(),
                       cellMemoryMatrix: cellMemoryMatrix.detached())
    
  }
  
  func backward(cache: LSTM<N>.Cache,
                previousCache: LSTM<N>.Cache?,
                activationOutputError: [[Tensor<N>.Scalar]],
                nextActivationError: [[Tensor<N>.Scalar]],
                nextCellError: [[Tensor<N>.Scalar]],
                batchSize: Int,
                parameters: Parameters) -> (inputs: Errors, weights: ParameterDerivatives, biases: ParameterDerivatives) {

    let activationError = activationOutputError + nextActivationError

    let lstm = cache.lstm
    let cellActivation = cache.cell
    let previousCellActivaion = previousCache?.cell ?? cache.cell.zerosLike() // if there's no previous state use 0s. Might need to come up with a better solution so we dont have to re-do this
    
    let tanActivationOfCellActivation = Tanh<N>().forward(tensor: cellActivation)
    
    // output gate error
    let oa = lstm.outputGate
    var eo = Tensor<N>(activationError) * tanActivationOfCellActivation
    eo = (eo * oa) * (1 - oa)
    
    // cell activation error
    var cellError = Tensor<N>(activationError) * oa
    cellError = cellError * self.device.derivate(tanActivationOfCellActivation, .tanh)
    cellError = cellError + Tensor<N>(nextCellError)
    
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
    
    var embedActivationError = ef.matmul(Tensor<N>(fgw.value.transpose2d()))
    embedActivationError = embedActivationError + (ei.matmul(Tensor<N>(igw.value.transpose2d())))
    embedActivationError = embedActivationError + (eo.matmul(Tensor<N>(ogw.value.transpose2d())))
    embedActivationError = embedActivationError + (eg.matmul(Tensor<N>(ggw.value.transpose2d())))
    
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
                                   embedding: Tensor<N>,
                                   activation: Tensor<N>,
                                   batchSize: Int) -> ParameterDerivatives {
    
    let transposed = embedding.concat(activation).value.transpose2d()
    
    let concat = Tensor<N>(transposed)
    
    let ef = lstmError.ef
    let ei = lstmError.ei
    let eo = lstmError.eo
    let eg = lstmError.eg
    
    var dfgw = concat.matmul(ef)
    var digw = concat.matmul(ei)
    var dogw = concat.matmul(eo)
    var dggw = concat.matmul(eg)

    if batchSize > 1 {
      dfgw = dfgw / Tensor<N>.Scalar(batchSize)
      digw = digw / Tensor<N>.Scalar(batchSize)
      dogw = dogw / Tensor<N>.Scalar(batchSize)
      dggw = dggw / Tensor<N>.Scalar(batchSize)
    }

    return .init(dForgetGate: dfgw,
                 dInputGate: digw,
                 dGateGate: dggw,
                 dOutputGate: dogw)
  }
}
