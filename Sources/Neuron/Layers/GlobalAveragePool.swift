//
//  GlobalAveragePool.swift
//
//

import Foundation
import NumSwift

public final class GlobalAvgPool: BaseLayer {
  public init(inputSize: TensorSize = TensorSize(array: [])) {
    super.init(inputSize: inputSize,
               biasEnabled: false,
               encodingType: .globalAvgPool)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type
  }
  
  override public func onInputSizeSet() {
    // outputSize will effectively 'flatten' with a global average at each channel
    outputSize = .init(rows: 1, columns: inputSize.depth, depth: 1)
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
    
    let tensorContext = TensorContext { [inputSize] inputs, gradient in
      // each input value at a given space only contributed 1 / (row * column) of the output
            
      let flatGradient = gradient.value.flatten()
      
      var wrtToInput = Tensor()
      
      for d in 0..<inputSize.depth {
        let gradientAtDepth = flatGradient[d] / (inputSize.rows.asTensorScalar * inputSize.columns.asTensorScalar)
        let value = Tensor.fillWith(value: gradientAtDepth,
                                    size: .init(rows: inputSize.rows,
                                                columns: inputSize.columns,
                                                depth: 1))
        wrtToInput = wrtToInput.concat(value, axis: 2)
      }
      
      return (wrtToInput, Tensor(), Tensor())
    }
    
    let outValue: [Tensor.Scalar] = tensor.value.compactMap { $0.flatten().mean }

    // forward calculation
    let out = Tensor(outValue, context: tensorContext)
    
    out.setGraph(tensor)
    
    return out
  }
}

