//
//  ResNet.swift
//
//

import Foundation
import NumSwift

public final class ResNet: BaseLayer {
  
  private var innerBlockSequential = Sequential()
  private var shortcutSequential = Sequential()
  private let outputRelu = ReLu()
  private let accumulator = GradientAccumulator()
  private let shortCutAccumulator = GradientAccumulator()

  private let filterCount: Int
  private let stride: Int
  
  private var shouldProjectInput: Bool {
    return filterCount != inputSize.depth || stride != 1
  }
  
  public init(inputSize: TensorSize? = nil,
              filterCount: Int,
              stride: Int = 1) {
    self.filterCount = filterCount
    self.stride = stride
    
    super.init(inputSize: inputSize,
               initializer: nil,
               biasEnabled: false,
               encodingType: .resNet)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type, filterCount, stride
  }

  override public func onInputSizeSet() {
    /// do something when the input size is set when calling `compile` on `Sequential`
    // build sequential?
    let initializer = self.initializer?.type ?? .heNormal
        
    innerBlockSequential.layers = [
      Conv2d(filterCount: filterCount,
             inputSize: inputSize,
             strides: (stride,stride),
             padding: .same,
             filterSize: (3,3),
             initializer: initializer),
      BatchNormalize(),
      ReLu(),
      Conv2d(filterCount: filterCount,
             strides: (1,1),
             padding: .same,
             filterSize: (3,3),
             initializer: initializer),
      BatchNormalize()
    ]
    
    shortcutSequential.layers = [
      Conv2d(filterCount: filterCount,
             inputSize: inputSize,
             strides: (stride,stride),
             padding: .same,
             filterSize: (1,1),
             initializer: initializer),
      BatchNormalize()
    ]
    
    innerBlockSequential.compile()
    shortcutSequential.compile()
    
    if shouldProjectInput {
      let outputSize = shortcutSequential.layers.last!.outputSize
      outputRelu.inputSize = outputSize
      self.outputSize = outputSize
    } else {
      outputSize = inputSize
      outputRelu.inputSize = inputSize
    }
    
  }
  
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let filterCount = try container.decodeIfPresent(Int.self, forKey: .filterCount) ?? 64
    let stride = try container.decodeIfPresent(Int.self, forKey: .stride) ?? 1

    self.init(filterCount: filterCount, stride: stride)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])

  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    let blockOut = innerBlockSequential(tensor, context: context)
    
    let tensorToAdd = if shouldProjectInput {
      // we project the input here to match the filter depth
      shortcutSequential(tensor, context: context)
    } else {
      tensor
    }
    
    // we need the graph from tensorToAdd here...
    // only `blockOut` is actually being backpropogated automatically because of how the autograd for addition works.
    let skipOut = blockOut + tensorToAdd
    let reLuOut = outputRelu.forward(tensor: skipOut)
    
    let tensorContext = TensorContext { [accumulator, shortCutAccumulator, shouldProjectInput, innerBlockSequential] inputs, gradient in
      // backpropogation calculation    // ∂F(x)/∂x -> output of ResNet block (without skip connection) wrt input
      // + 1 -> is the skip connection because we're just adding the inputs back to the output the partial gradient wrt to the input is just 1.
      // ∇y × (∂F(x)/∂x + 1)
      
      // backprop all the way through the the sequentials because the graphs are built automatically for us
      let reluGradients = reLuOut.gradients(delta: gradient)
      
      let wrtInputs = reluGradients.input[safe: 0, Tensor()] + 1
      
      let blockGradientsInput = Array(reluGradients.input[0..<innerBlockSequential.layers.count])
      let blockGradientsWeights = Array(reluGradients.weights[0..<innerBlockSequential.layers.count])
      let blockGradientsBiases = Array(reluGradients.biases[0..<innerBlockSequential.layers.count])

      accumulator.insert(.init(input: blockGradientsInput,
                               weights: blockGradientsWeights,
                               biases: blockGradientsBiases))
      
      if shouldProjectInput {
        let shortCutGradientsInput = Array(reluGradients.input[innerBlockSequential.layers.count...])
        let shortCutGradientsWeights = Array(reluGradients.weights[innerBlockSequential.layers.count...])
        let shortCutGradientsBiases = Array(reluGradients.biases[innerBlockSequential.layers.count...])
        
        shortCutAccumulator.insert(.init(input: shortCutGradientsInput,
                                         weights: shortCutGradientsWeights,
                                         biases: shortCutGradientsBiases))
      }
      
      return (wrtInputs, Tensor(), Tensor())
    }
    
    let out = Tensor(reLuOut.value, context: tensorContext)
    
    out.setGraph(tensor)
    
    // forward calculation
    return out
  }
  
  public override func apply(gradients: (weights: Tensor, biases: Tensor), learningRate: Tensor.Scalar) {
    let sequentialGradients = accumulator.accumulate()
    
    innerBlockSequential.apply(gradients: sequentialGradients, learningRate: learningRate)
    
    if shouldProjectInput {
      let shortCutGradients = shortCutAccumulator.accumulate()
      shortcutSequential.apply(gradients: shortCutGradients, learningRate: learningRate)
    }
    
    accumulator.clear()
  }
}

