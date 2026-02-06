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
  private var training: Bool = true
  
  public override var isTraining: Bool {
    get {
      super.isTraining
    }
    set {
      innerBlockSequential.isTraining = newValue
      shortcutSequential.isTraining = newValue
    }
  }
  
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
    
    innerBlockSequential.name = "ResNet-MainPath"
    shortcutSequential.name = "ResNet-ShortcutPath"
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type, filterCount, stride, innerBlockSequential, shortcutSequential
  }
  
  override public func onBatchSizeSet() {
    innerBlockSequential.batchSize = batchSize
    shortcutSequential.batchSize = batchSize
  }
  
  override public func onInputSizeSet() {
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
        BatchNormalize(gamma: Array(repeating: 0.0, count: filterCount))
      ]
    }
    
    if shortcutSequential.layers.isEmpty {
      shortcutSequential.layers = [
        Conv2d(filterCount: filterCount,
               inputSize: inputSize,
               strides: (stride,stride),
               padding: .same,
               filterSize: (1,1),
               initializer: initializer)
      ]
    }
    
    innerBlockSequential.compile()
    shortcutSequential.compile()
    
    let reluInputSize = if shouldProjectInput {
      shortcutSequential.layers.last?.outputSize ?? inputSize
    } else {
      inputSize
    }
    
    outputRelu.inputSize = reluInputSize
    outputSize = reluInputSize
    
    onBatchSizeSet()
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
  
  public override func forward(tensorBatch: TensorBatch, context: NetworkContext) -> TensorBatch {
    // we need to pass the full batch to BatchNormalize to calculate the global mean on each batch. Just like what happens when it's outside of the optimizer
    let detachedInputs = tensorBatch.map { $0.detached() }
    let blockOuts = innerBlockSequential.predict(batch: detachedInputs, context: context)
    
    // set a copied input so we can separate the input tensors of the two paths.
    // this allows us to pull the gradients wrt to each input
    let copiedInputTensors = tensorBatch.map { $0.copy() }
    
    let tensorToAdd = if shouldProjectInput {
      // we project the input here to match the filter depth
      shortcutSequential.predict(batch: copiedInputTensors, context: context)
    } else {
      copiedInputTensors
    }
    
    let skipOut = blockOuts + tensorToAdd
    
    let reLuOuts = outputRelu.forward(tensorBatch: skipOut, context: context)
    
    var outs: TensorBatch = []
    
    for (i, reLuOut) in reLuOuts.enumerated() {
      let tensor = tensorBatch[i]
      let detachedInput = detachedInputs[i]
      let copiedInputTensor = copiedInputTensors[i]
      
      let out = buildForward(input: tensor,
                             reLuOut: reLuOut,
                             detachedInput: detachedInput,
                             copiedInputTensor: copiedInputTensor)
      outs.append(out)
    }
    
    return outs
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    // we detach the input tensor because we want to stop at this tensor in respect to this sequential
    // not all the way up the graph to possibly other input layers
    let detachedInput = tensor.detached()
    detachedInput.label = "detachedInput"
    let blockOut = innerBlockSequential.predict(batch: [detachedInput], context: .init())[safe: 0, Tensor()]
    blockOut.label = "blockOut"
    
    // set a copied input so we can separate the input tensors of the two paths.
    // this allows us to pull the gradients wrt to each input
    let copiedInputTensor = tensor.copy()
    copiedInputTensor.label = "copiedInputTensor"
    
    let tensorToAdd = if shouldProjectInput {
      // we project the input here to match the filter depth
      shortcutSequential.predict(batch: [copiedInputTensor], context: .init())[safe: 0, Tensor()]
    } else {
      copiedInputTensor
    }
    
    tensorToAdd.label = "tensorToAdd"
    
    let skipOut = blockOut + tensorToAdd
    let reLuOut = outputRelu.forward(tensor: skipOut)
    
    let out = buildForward(input: tensor,
                           reLuOut: reLuOut,
                           detachedInput: detachedInput,
                           copiedInputTensor: copiedInputTensor)
    
    // forward calculation
    return out
  }
  
  public override func apply(gradients: (weights: Tensor, biases: Tensor), learningRate: Tensor.Scalar) {
    
    let sequentialGradients = accumulator.accumulate(clearAtEnd: true)
    innerBlockSequential.apply(gradients: sequentialGradients, learningRate: learningRate)
    
    if shouldProjectInput {
      let shortCutGradients = shortCutAccumulator.accumulate(clearAtEnd: true)
      shortcutSequential.apply(gradients: shortCutGradients, learningRate: learningRate)
    }
  }
  
  // MARK: - Private
  
  private func buildForward(input: Tensor, reLuOut: Tensor, detachedInput: Tensor, copiedInputTensor: Tensor) -> Tensor {
    let tensorContext = TensorContext { [accumulator, shortCutAccumulator, shouldProjectInput, innerBlockSequential, shortcutSequential, reLuOut] inputs, gradient, wrt in
      
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
      
      let gradientToAdd = if shouldProjectInput {
        reluGradientsWrtProjectedInput.input[safe: 0, Tensor()]
      } else {
        gradient
      }
      
      let wrtInputs = reluGradients.input[safe: 0, Tensor()] + gradientToAdd // add gradient FROM skip connection. Direct path for gradients
      return (wrtInputs, Tensor(), Tensor())
    }
    
    let out = Tensor(Tensor.Value(reLuOut.storage), size: reLuOut.size, context: tensorContext)
    
    out.setGraph(input)
    
    return out
  }
}

