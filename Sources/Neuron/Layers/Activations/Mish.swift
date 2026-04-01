import Foundation
import NumSwift

//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/4/22.
//

import Foundation
import NumSwift

public final class Mish: BaseActivationLayer {
  public init(inputSize: TensorSize = TensorSize(array: []),
              initializer: InitializerType = .heNormal,
              linkId: String = UUID().uuidString) {
    
    super.init(inputSize: inputSize,
               type: .mish,
               linkId: linkId,
               encodingType: .mish)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type, linkId
  }
  
  convenience public required init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    
    self.outputSize = inputSize
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(linkId, forKey: .linkId)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    let forward = tensor.storage
    let newStorage = TensorStorage.create(count: forward.count)
    let tanhForward = TensorStorage.create(count: forward.count)
    
    for i in 0..<newStorage.count {
      let value = forward[i]
      let tanhCalc = Tensor.Scalar.tanh(Tensor.Scalar.log(1 + Tensor.Scalar.exp(value)))
      tanhForward[i] = tanhCalc
      newStorage[i] = value * tanhCalc
    }
    
    let tensorContext = TensorContext { inputs, gradient, wrt in
      // backpropogation calculation
      
      let tanhSp = Tensor(storage: tanhForward, size: self.inputSize)
      
      let sigmoid = Sigmoid(inputSize: self.inputSize)
      let sigmoidOut = sigmoid(inputs).detached()
      
      let sech2 = (1 - tanhSp * tanhSp)
      
      let wrtInput = (tanhSp + inputs * sigmoidOut * sech2) * gradient
      wrtInput.label = "mish_input_gradient"
    
      return (wrtInput, Tensor(), Tensor())
    }
    
    // forward calculation - setGraph connects `tensor` so the custom context fires during backprop
    let out = Tensor(storage: newStorage, size: outputSize, context: tensorContext)
    out.label = "mish"
    out.setGraph(tensor)
    
    return out
  }
}

