//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/4/22.
//

import Foundation
import NumSwift

public final class PReLu: BaseActivationLayer {
  
  public override var weights: Tensor {
    get {
      Tensor(alpha)
    }
    set {
      alpha = newValue.asScalar()
    }
  }
  
  private var alpha: Tensor.Scalar = 0.25

  public init(inputSize: TensorSize = TensorSize(array: []),
              initializer: InitializerType = .heNormal,
              linkId: String = UUID().uuidString) {
    
    super.init(inputSize: inputSize,
               type: .prelu,
               linkId: linkId,
               encodingType: .prelu)
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
  
  /// Encodes the layer's configuration into the given encoder.
  /// - Parameter encoder: The encoder to write layer data into.
  /// - Throws: An error if any value fails to encode.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(linkId, forKey: .linkId)
  }

  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    let forward = tensor.storage
    let newStorage = TensorStorage.create(count: forward.count)
    
    for i in 0..<newStorage.count {
      let value = forward[i]
      
      let calc = if value <= 0 {
        alpha * value
      } else {
        value
      }
      
      newStorage[i] = calc
    }
    
    let tensorContext = TensorContext { [alpha] inputs, gradient, wrt in
      let wrtInputStorage = TensorStorage.create(count: inputs.storage.count)
      var wrtToAlpha: Tensor.Scalar = 0
      
      for i in 0..<wrtInputStorage.count {
        let value = inputs.storage[i]
        let gradientValue = gradient.storage[i]
        
        let calc: Tensor.Scalar = if value > 0 {
          1
        } else {
          alpha
        }
        
        if value < 0 {
          wrtToAlpha += gradientValue * value
        }
        
        wrtInputStorage[i] = gradientValue * calc
      }
      
      let wrtInput = Tensor(storage: wrtInputStorage, size: gradient.size)
      
      return (wrtInput, Tensor(wrtToAlpha), Tensor())
    }
    
    // forward calculation - setGraph connects `tensor` so the custom context fires during backprop
    let out = Tensor(storage: newStorage, size: outputSize, context: tensorContext)
    out.label = "prelu"
    out.setGraph(tensor)
    
    return out
  }
  
  public override func apply(gradients: (weights: Tensor, biases: Tensor), learningRate: Tensor.Scalar) {
    let weightScalar = gradients.weights.asScalar()
    
    alpha = alpha - learningRate * weightScalar
  }
}

