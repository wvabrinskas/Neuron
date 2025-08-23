//
//  ResNet.swift
//
//

import Foundation
import NumSwift

public final class ResNet: BaseLayer {
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               initializer: nil,
               biasEnabled: false,
               encodingType: .resNet)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type
  }
  
  private var innerBlockSequential = Sequential()
  private let outputRelu = ReLu()
  
  override public func onInputSizeSet() {
    /// do something when the input size is set when calling `compile` on `Sequential`
    outputSize = inputSize
    // build sequential?
    
    innerBlockSequential.layers = [
      Conv2d(filterCount: 64,
             inputSize: inputSize,
             strides: (1,1),
             padding: .same,
             filterSize: (3,3),
             initializer: .heNormal),
      BatchNormalize(),
      ReLu(),
      Conv2d(filterCount: 64,
             inputSize: inputSize,
             strides: (1,1),
             padding: .same,
             filterSize: (3,3),
             initializer: .heNormal),
      BatchNormalize()
    ]
    
    outputRelu.inputSize = inputSize
    
    innerBlockSequential.compile()
  }
  
  convenience public required init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    let tensorContext = TensorContext { inputs, gradient in
      // backpropogation calculation
      return (Tensor(), Tensor(), Tensor())
    }
    
    let blockOut = innerBlockSequential(tensor, context: context)
    let skipOut = blockOut + tensor
    let reLuOut = outputRelu.forward(tensor: skipOut)
    
    // forward calculation
    return reLuOut
  }
  
  private func backwards() {
    // ∂F(x)/∂x -> output of ResNet block (without skip connection) wrt input
    // + 1 -> is the skip connection because we're just adding the inputs back to the output the partial gradient wrt to the input is just 1.
    // ∇y × (∂F(x)/∂x + 1)
    
  }
}

