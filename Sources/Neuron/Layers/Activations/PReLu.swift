//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/4/22.
//

import Foundation
import NumSwift

/// Performs a Parametric ReLU (PReLU) activation.
///
/// PReLU is a variant of Leaky ReLU where the negative-slope coefficient `alpha`
/// is a learnable parameter rather than a fixed hyperparameter. The activation
/// computes `x` for positive inputs and `alpha * x` for negative inputs, with
/// `alpha` updated during training via the standard gradient descent step.
public final class PReLu: BaseActivationLayer {

  /// Exposes the learnable negative-slope coefficient `alpha` as a single-element tensor.
  ///
  /// Getting wraps the current `alpha` in a scalar tensor; setting unpacks the first
  /// scalar of the tensor into `alpha`. This conforms to the `Layer` weights contract
  /// so that optimizers can apply gradient updates.
  public override var weights: Tensor {
    get {
      Tensor(alpha)
    }
    set {
      alpha = newValue.asScalar()
    }
  }

  private var alpha: Tensor.Scalar = 0.25

  /// Default initializer for a PReLU activation.
  ///
  /// - Parameters:
  ///   - inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  ///   - initializer: Weight initializer type. Retained for API symmetry; `alpha` is initialized to `0.25` regardless.
  ///   - linkId: A unique string identifier for this layer. Defaults to a new UUID string.
  public init(inputSize: TensorSize = TensorSize(array: []),
              initializer: InitializerType = .heNormal,
              linkId: String = UUID().uuidString) {
    
    super.init(inputSize: inputSize,
               type: .prelu,
               linkId: linkId,
               encodingType: .prelu)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type, linkId, alpha
  }

  convenience public required init(from decoder: Decoder) throws {
    self.init()
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    self.alpha = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .alpha) ?? 0.25
    
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
    try container.encode(alpha, forKey: .alpha)
  }

  /// Applies the PReLU activation element-wise.
  ///
  /// For each element `x`: returns `x` when `x > 0` and `alpha * x` otherwise.
  /// Backpropagation produces gradients both with respect to the input and the
  /// learnable `alpha` parameter (accumulated only over negative inputs).
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Activated tensor wired into the computation graph.
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
  
  /// Updates the learnable `alpha` coefficient using gradient descent.
  ///
  /// - Parameters:
  ///   - gradients: Tuple containing the scalar weight gradient (w.r.t. `alpha`) and biases (unused).
  ///   - learningRate: Step size applied to the gradient update.
  public override func apply(gradients: (weights: Tensor, biases: Tensor), learningRate: Tensor.Scalar) {
    let weightScalar = gradients.weights.asScalar()
    
    alpha = alpha - learningRate * weightScalar
  }
}

