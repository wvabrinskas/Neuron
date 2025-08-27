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
  
  public override var weights: Tensor {
    get {
      let innerBlockWeights = (try? innerBlockSequential.exportWeights()) ?? []
      var flatInnerBlockWeights: [Tensor] = innerBlockWeights.fullFlatten()
      
      if shouldProjectInput {
        let shortCutSequentialWeights = (try? shortcutSequential.exportWeights()) ?? []
        let flatShortCutSequentialWeights: [Tensor] = shortCutSequentialWeights.fullFlatten()
        flatInnerBlockWeights.append(contentsOf: flatShortCutSequentialWeights)
      }
      
      var firstTensor = flatInnerBlockWeights.removeFirst()
      
      flatInnerBlockWeights.forEach { t in
        firstTensor = firstTensor.concat(t, axis: 2)
      }
    
      return firstTensor
    }
    
    set {
      let newWeights = newValue
      
      let innerBlockCount = innerBlockSequential.layers.count
      let shortCutSequentialCount = shortcutSequential.layers.count
      
      let totalCount = if shouldProjectInput {
        innerBlockCount + shortCutSequentialCount
      } else {
        innerBlockCount
      }

      // todo: figure out how to apply weights
    }
  }
  
  private var shouldProjectInput: Bool {
    return filterCount != inputSize.depth || stride != 1
  }
  
  public init(inputSize: TensorSize? = nil,
              initializer: InitializerType = Constants.defaultInitializer,
              filterCount: Int,
              stride: Int = 1) {
    self.filterCount = filterCount
    self.stride = stride
    
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: false,
               encodingType: .resNet)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type, filterCount, stride, innerBlockSequential, shortcutSequential
  }

  override public func onInputSizeSet() {
    /// do something when the input size is set when calling `compile` on `Sequential`
    // build sequential?
    let initializer = initializer.type
        
    if innerBlockSequential.layers.isEmpty {
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
    }
    
    if shortcutSequential.layers.isEmpty {
      shortcutSequential.layers = [
        Conv2d(filterCount: filterCount,
               inputSize: inputSize,
               strides: (stride,stride),
               padding: .same,
               filterSize: (1,1),
               initializer: initializer),
        BatchNormalize()
      ]
    }
    
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
        
    let innerBlockSequential = try container.decodeIfPresent(Sequential.self, forKey: .innerBlockSequential) ?? Sequential()
    let shortcutSequential = try container.decodeIfPresent(Sequential.self, forKey: .shortcutSequential) ?? Sequential()
    
    self.innerBlockSequential = innerBlockSequential
    self.shortcutSequential = shortcutSequential
    
    // set input size AFTER setting sequentials because we need onInputSizeSet() to compile them
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(filterCount, forKey: .filterCount)
    try container.encode(stride, forKey: .stride)
    try container.encode(innerBlockSequential, forKey: .innerBlockSequential)
    try container.encode(shortcutSequential, forKey: .shortcutSequential)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    // we detach the input tensor because we want to stop at this tensor in respect to this sequential
    // not all the way up the graph to possibly other input layers
    let detachedInput = tensor.detached()
    let blockOut = innerBlockSequential(detachedInput, context: context)
    
    // set a copied input so we can separate the input tensors of the two paths.
    // this allows us to pull the gradients wrt to each input
    let copiedInputTensor = tensor.copy()

    let tensorToAdd = if shouldProjectInput {
      // we project the input here to match the filter depth
      shortcutSequential(copiedInputTensor, context: context)
    } else {
      copiedInputTensor
    }

    let skipOut = blockOut + tensorToAdd
    let reLuOut = outputRelu.forward(tensor: skipOut)
    
    let tensorContext = TensorContext { [accumulator, shortCutAccumulator, shouldProjectInput, innerBlockSequential, shortcutSequential] inputs, gradient in
            
      // backprop all the way through the the sequentials because the graphs are built automatically for us
      let reluGradients = reLuOut.gradients(delta: gradient, wrt: detachedInput)
      let reluGradientsWrtProjectedInput = reLuOut.gradients(delta: gradient, wrt: copiedInputTensor)
      
      let blockGradientsWeights = Array(reluGradients.weights[0..<innerBlockSequential.layers.count])
      let blockGradientsBiases = Array(reluGradients.biases[0..<innerBlockSequential.layers.count])

      accumulator.insert(.init(input: [], // we dont need these
                               weights: blockGradientsWeights,
                               biases: blockGradientsBiases))
      
      if shouldProjectInput {
        let shortCutGradientsWeights = Array(reluGradientsWrtProjectedInput.weights[0..<shortcutSequential.layers.count])
        let shortCutGradientsBiases = Array(reluGradientsWrtProjectedInput.biases[0..<shortcutSequential.layers.count])

        shortCutAccumulator.insert(.init(input: [], // we dont need these
                                         weights: shortCutGradientsWeights,
                                         biases: shortCutGradientsBiases))
        
      }
      
      // because the input is used in two paths we can just sum the gradients...
      let wrtInputs = reluGradients.input[safe: 0, Tensor()] + 1 // + 1 because wrt to the input skip path is 1
      
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
    accumulator.clear()

    if shouldProjectInput {
      let shortCutGradients = shortCutAccumulator.accumulate()
      shortcutSequential.apply(gradients: shortCutGradients, learningRate: learningRate)
      shortCutAccumulator.clear()
    }
  }
}

