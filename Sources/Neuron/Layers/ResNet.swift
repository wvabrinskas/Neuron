//
//  ResNet.swift
//
//

import Foundation
import NumSwift

/// A residual network (ResNet) block layer that applies a skip connection around an inner block of operations.
///
/// This layer implements the residual learning framework, where the input is added to the output
/// of the inner block sequence, optionally projecting the input via a shortcut sequential when
/// the dimensions do not match.
public final class ResNet: BaseLayerGroup {
  
  private var shortcutSequential = Sequential()
  private let outputRelu = ReLu()
  
  private let filterCount: Int
  private let stride: Int
  private var training: Bool = true
  
  /// A Boolean value indicating whether the layer is in training mode.
  ///
  /// Setting this property propagates the training state to both the inner block
  /// sequential and the shortcut sequential.
  public override var isTraining: Bool {
    get {
      super.isTraining
    }
    set {
      innerBlockSequential.isTraining = newValue
      shortcutSequential.isTraining = newValue
    }
  }
  
  /// The concatenated weights of the inner block and, if input projection is required, the shortcut sequential.
  ///
  /// Getting this property exports and flattens weights from the inner block sequential,
  /// appending the shortcut sequential weights when input projection is enabled, and
  /// concatenates them into a single `Tensor` along axis 2.
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
  
  /// Creates a residual block with an optional projection shortcut.
  ///
  /// - Parameters:
  ///   - inputSize: Optional known input shape.
  ///   - initializer: Initializer used by internal convolution layers.
  ///   - filterCount: Number of output channels for the block.
  ///   - stride: Spatial stride for the first convolution/projection path.
  public init(inputSize: TensorSize? = nil,
              initializer: InitializerType = Constants.defaultInitializer,
              filterCount: Int,
              stride: Int = 1) {
    self.filterCount = filterCount
    self.stride = stride
    
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: false,
               encodingType: .resNet,
               layers: { inputSize in
      [
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
    })
    
    innerBlockSequential.name = "ResNet-MainPath"
    shortcutSequential.name = "ResNet-ShortcutPath"
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type, filterCount, stride, innerBlockSequential, shortcutSequential
  }
  
  override public func onBatchSizeSet() {
    super.onBatchSizeSet()
    shortcutSequential.batchSize = batchSize
  }
  
  override public func onInputSizeSet() {
    let initializer = initializer.type
    
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
    
    shortcutSequential.compile()
    
    let reluInputSize = if shouldProjectInput {
      shortcutSequential.layers.last?.outputSize ?? inputSize
    } else {
      inputSize
    }
    
    outputRelu.inputSize = reluInputSize
    outputSize = reluInputSize
    
    super.onInputSizeSet()
    
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
  
  /// Encodes residual block configuration and internal path networks.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(filterCount, forKey: .filterCount)
    try container.encode(stride, forKey: .stride)
    try container.encode(innerBlockSequential, forKey: .innerBlockSequential)
    try container.encode(shortcutSequential, forKey: .shortcutSequential)
  }
  
  /// Runs the residual block forward pass for a batch.
  ///
  /// - Parameters:
  ///   - tensorBatch: Input batch.
  ///   - context: Batch/thread metadata.
  /// - Returns: Batch outputs after main path, shortcut add, and output ReLU.
  public override func forward(tensorBatch: TensorBatch, context: NetworkContext) -> TensorBatch {
    // we need to pass the full batch to BatchNormalize to calculate the global mean on each batch. Just like what happens when it's outside of the optimizer
    let detachedInputs = tensorBatch.map { $0.detached() }
    let blockOuts = super.forward(tensorBatch: detachedInputs, context: context)
    
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
  
  /// Runs the residual block forward pass for a single tensor.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Residual block output tensor.
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    // we detach the input tensor because we want to stop at this tensor in respect to this sequential
    // not all the way up the graph to possibly other input layers
    let detachedInput = tensor.detached()
    let blockOut = super.forward(tensor: detachedInput, context: context)
    
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
  
  /// Applies aggregated gradients to inner and shortcut subpaths.
  ///
  /// - Parameters:
  ///   - gradients: Combined weight and bias gradients from backward pass.
  ///   - learningRate: Learning rate used by sub-layer update paths.
  public override func apply(gradients: (weights: Tensor, biases: Tensor), learningRate: Tensor.Scalar) {
    // this isn't using the optimizer gradients at all...
        
    guard gradients.weights.isEmpty == false else {
      return
    }
    
    applyGradients(gradients: gradients,
                   learningRate: learningRate)
  }
  
  // MARK: - Private
  
  private func applyGradients(gradients: (weights: Tensor, biases: Tensor), learningRate: Tensor.Scalar) {
    let (currentWeightOffset, currentBiasOffset) = super.applyGradientsToInnerBlock(gradients: gradients, learningRate: learningRate)
    
    applyGradientsToShortcutBlock(gradients: gradients,
                                  offsets: (currentWeightOffset, currentBiasOffset),
                                  learningRate: learningRate)
  }
  
  private func applyGradientsToShortcutBlock(gradients: (weights: Tensor, biases: Tensor),
                                             offsets: (weights: Int, biases: Int),
                                             learningRate: Tensor.Scalar) {
    guard shouldProjectInput else { return }
    
    let shortcutWeightOffset = offsets.weights
    let shortcutBiasOffset = offsets.biases
    
    var weightGradients: [Tensor] = []
    var biasGradients: [Tensor] = []
    
    var lastTotalWeights: Int = 0
    var lastTotalBiases: Int = 0
        
    for layer in shortcutSequential.layers {
      guard layer.usesOptimizer else {
        // add placeholder for layers that don't need gradients
        weightGradients.append(Tensor())
        biasGradients.append(Tensor())
        continue
      }
      
      // this should be correct given how we calculate weight size at each layer
      let size = layer.weights.size
      let totalWeights = size.rows * size.columns * size.depth
      let indexOffset = lastTotalWeights + shortcutWeightOffset
      
      let biasSize = layer.biases.size
      let totalBiases = biasSize.rows * biasSize.columns * biasSize.depth
      let indexBiasOffset = lastTotalBiases + shortcutBiasOffset
      
      let layerWeights = Tensor(Tensor.Value(gradients.weights.storage[indexOffset..<indexOffset + totalWeights]), size: size)
      let layerBiases = Tensor(Tensor.Value(gradients.biases.storage[indexBiasOffset..<indexBiasOffset + totalBiases]), size: biasSize)
      
      weightGradients.append(layerWeights)
      biasGradients.append(layerBiases)
      
      lastTotalBiases += totalBiases
      lastTotalWeights += totalWeights
    }
    
    shortcutSequential.apply(gradients: .init(input: [],
                                              weights: weightGradients,
                                              biases: biasGradients),
                             learningRate: learningRate)
    
  }
  
  private func buildForward(input: Tensor, reLuOut: Tensor, detachedInput: Tensor, copiedInputTensor: Tensor) -> Tensor {
    let tensorContext = TensorContext { [shouldProjectInput, innerBlockSequential, shortcutSequential, reLuOut] inputs, gradient, wrt in
      
      let reluGradients = reLuOut.gradients(delta: gradient, wrt: detachedInput)

      let reluGradientsWrtProjectedInput = reLuOut.gradients(delta: gradient, wrt: copiedInputTensor)
            
      var weightGradients = Array(reluGradients.weights[0..<innerBlockSequential.layers.count]).flatMap { $0.storage }
      
      var biasGradients = Array(reluGradients.biases[0..<innerBlockSequential.layers.count]).flatMap { $0.storage }
      
      if shouldProjectInput {
        let shortCutGradientsWeights = Array(reluGradientsWrtProjectedInput.weights[0..<shortcutSequential.layers.count])
        let shortCutGradientsBiases = Array(reluGradientsWrtProjectedInput.biases[0..<shortcutSequential.layers.count])
        
        weightGradients.append(contentsOf: shortCutGradientsWeights.flatMap(\.storage))
        biasGradients.append(contentsOf: shortCutGradientsBiases.flatMap(\.storage))
      }
      
      let gradientToAdd = if shouldProjectInput {
        reluGradientsWrtProjectedInput.input[safe: 0, Tensor()]
      } else {
        gradient
      }
      
      let wrtInputs = reluGradients.input[safe: 0, Tensor()] + gradientToAdd // add gradient FROM skip connection. Direct path for gradients
      return (wrtInputs, Tensor(weightGradients), Tensor(biasGradients))
    }
    
    let out = Tensor(reLuOut.storage, size: reLuOut.size, context: tensorContext)
    
    out.setGraph(input)
    
    return out
  }
}

