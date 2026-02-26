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

public final class RexNet: BaseLayerGroup {
  private let expandRatio: Tensor.Scalar
  private let strides: (rows: Int, columns: Int)
  private let outChannels: Int
  
  private var shouldSkip: Bool {
    strides.columns == 1 && strides.rows == 1 && outChannels == inputSize.depth
  }
  
  public init(inputSize: TensorSize? = nil,
              initializer: InitializerType = Constants.defaultInitializer,
              strides: (rows: Int, columns: Int) = (1,1),
              outChannels: Int,
              expandRatio: Tensor.Scalar = 0) {
    
    self.expandRatio = expandRatio
    self.strides = strides
    self.outChannels = outChannels
    
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: false,
               encodingType: .rexNet) { inputSize in
      var layers: [Layer] = []
      
      let expanded = inputSize.depth.asTensorScalar * expandRatio
      
      // Step 1: Expand (skip if expand_ratio == 1)
      if expandRatio > 1 {
        let expandLayers: [Layer] = [
          Conv2d(filterCount: Int(expanded),
                 inputSize: inputSize,
                 strides: (1,1),
                 padding: .valid,
                 filterSize: (1,1),
                 initializer: initializer,
                 biasEnabled: false),
          BatchNormalize(),
          Swish()
        ]
        
        layers.append(contentsOf: expandLayers)
      }
      
      // Step 2: Depthwise spatial conv
      let depthWiseLayers: [Layer] = [
        DepthwiseConv2d(inputSize: layers.isEmpty ? inputSize : nil,
                        strides: strides,
                        padding: .same,
                        filterSize: (3,3),
                        initializer: initializer,
                        biasEnabled: false),
        BatchNormalize(),
        Swish()
      ]
      
      layers.append(contentsOf: depthWiseLayers)
      
      // Step 3: Squeeze-and-Excite (optional)
      
      // Step 4: Project back down to out_channels
      let projectLayers: [Layer] = [
        Conv2d(filterCount: outChannels,
               strides: (1,1),
               padding: .valid,
               filterSize: (1,1),
               initializer: initializer,
               biasEnabled: false),
        BatchNormalize()
      ]
      
      layers.append(contentsOf: projectLayers)
      
      return layers
    }
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    /// do something when the input size is set when calling `compile` on `Sequential`
    /// like setting the output size or initializing the weights
    outputSize = innerBlockSequential.layers.last?.outputSize ?? inputSize
  }
  
  convenience public required init(from decoder: Decoder) throws {
    self.init(outChannels: 1)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  public override func forward(tensorBatch: TensorBatch, context: NetworkContext) -> TensorBatch {
    var result: TensorBatch = []
    
    for tensor in tensorBatch {
      result.append(forward(tensor: tensor, context: context))
    }
    
    return result
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    
    let forwardPass = if shouldSkip {
      super.forward(tensor: tensor, context: context) + tensor
    } else {
      super.forward(tensor: tensor, context: context)
    }
    
    let context = TensorContext { inputs, gradient, wrt in
      // backpropogation calculation
      
      let mainFlowInputGradients = forwardPass.gradients(delta: gradient, wrt: inputs)
      
      let wrtInput = if self.shouldSkip {
        mainFlowInputGradients.input[safe: 0, Tensor()] + gradient
      } else {
        mainFlowInputGradients.input[safe: 0, Tensor()]
      }
      
      let wrtWeights: Tensor = Tensor(mainFlowInputGradients.weights.flatMap(\.storage))
      let wrtBiases: Tensor = Tensor(mainFlowInputGradients.biases.flatMap(\.storage))

      return (wrtInput,
              wrtWeights,
              wrtBiases)
    }
    
    // forward calculation
    let out = Tensor(forwardPass.storage,
                     size: forwardPass.size,
                     context: context)
    
    out.label = "RexNet"
    out.setGraph(tensor)
    
    return out
  }
}
