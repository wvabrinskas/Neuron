//
//  LayerGroup.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/25/26.
//

import NumSwift
import Foundation

/// A protocol that combines `Layer` and `BaseLayer` conformance for grouped layer structures
/// that contain an inner sequential block of sub-layers.
public protocol LayerGrouping: Layer, BaseLayer {
  /// The sequential container that holds the inner block of sub-layers managed by this group.
  var innerBlockSequential: Sequential { get }
}


/// A base class for layer groups that manage an inner sequential block of sub-layers,
/// supporting forward passes, weight export, and gradient application across grouped layers.
public class BaseLayerGroup: BaseLayer, LayerGrouping {
  
  /// A type alias representing the offsets into gradient arrays for weights and biases
  /// within a layer group's inner block.
  public typealias GradientOffsets = (weights: Int, biases: Int)

  /// The sequential container holding the inner block of sub-layers for this layer group.
  public var innerBlockSequential: Sequential = .init()
  
  private var layers: (_ inputSize: TensorSize) -> [Layer]
  
  /// A Boolean indicating whether the layer group is in training mode.
  ///
  /// Setting this value propagates the training state to the inner sequential block.
  public override var isTraining: Bool {
    get {
      super.isTraining
    }
    set {
      super.isTraining = newValue
      innerBlockSequential.isTraining = newValue
    }
  }

  /// The concatenated weights of all layers within the inner sequential block.
  ///
  /// Getting this value exports and flattens weights from the inner block, concatenating
  /// them along axis 2 into a single `Tensor`. Setting is currently a no-op.
  public override var weights: Tensor {
    get {
      let innerBlockWeights = (try? innerBlockSequential.exportWeights()) ?? []
      var flatInnerBlockWeights: [Tensor] = innerBlockWeights.fullFlatten()
      var firstTensor = flatInnerBlockWeights.removeFirst()
      
      flatInnerBlockWeights.forEach { t in
        firstTensor = firstTensor.concat(t, axis: 2)
      }
      
      let weightTensor = Tensor(storage: firstTensor.storage.copy(),
                                size: .init(rows: 1,
                                            columns: firstTensor.storage.count,
                                            depth: 1))
      
      return weightTensor
    }
    set {
      // todo: figure out how to apply weights
    }
  }
  
  /// Initializes a `BaseLayerGroup` with the given configuration and a closure that builds
  /// the inner sub-layers based on the resolved input size.
  ///
  /// - Parameters:
  ///   - inputSize: The optional input tensor size for the layer group.
  ///   - initializer: The weight initializer type to use for sub-layers.
  ///   - biasEnabled: Whether bias terms are enabled. Defaults to `false`.
  ///   - linkId: A unique string identifier for this layer. Defaults to a new UUID string.
  ///   - encodingType: The encoding strategy used for this layer group.
  ///   - layers: A closure that receives the resolved input size and returns the array of sub-layers.
  public init(inputSize: TensorSize?,
              initializer: InitializerType,
              biasEnabled: Bool = false,
              linkId: String = UUID().uuidString,
              encodingType: EncodingType,
              layers: @escaping (_ inputSize: TensorSize) -> [Layer]) {
    
    self.layers = layers
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: biasEnabled,
               linkId: linkId,
               encodingType: encodingType)
  }
  
  /// Not supported on `BaseLayerGroup` — subclasses must override with their own serialization logic.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: Always triggers a `fatalError`; implement in concrete subclasses.
  required convenience public init(from decoder: Decoder) throws {
    fatalError("init(from:) has not been implemented")
  }
  
  override public func onInputSizeSet() {
    if innerBlockSequential.layers.isEmpty {
      innerBlockSequential.layers = layers(inputSize)
    }
    
    innerBlockSequential.compile()
    onBatchSizeSet()
  }
  
  override public func onBatchSizeSet() {
    innerBlockSequential.batchSize = batchSize
  }
  
  /// Runs the residual block forward pass for a single tensor.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Residual block output tensor.
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    forward(tensorBatch: [tensor], context: context)[safe: 0, Tensor()]
  }
  
  /// Applies the given weight and bias gradients to the inner sequential block.
  ///
  /// - Parameters:
  ///   - gradients: A tuple containing weight and bias gradient tensors.
  ///   - learningRate: The scalar learning rate to use when applying gradients.
  public override func apply(gradients: (weights: Tensor, biases: Tensor), learningRate: Tensor.Scalar) {
    guard gradients.weights.isEmpty == false else {
      return
    }
    
    applyGradientsToInnerBlock(gradients: gradients, learningRate: learningRate)
  }
  
  /// Runs the residual block forward pass for a batch.
  ///
  /// - Parameters:
  ///   - tensorBatch: Input batch.
  ///   - context: Batch/thread metadata.
  /// - Returns: Batch outputs after main path, shortcut add, and output ReLU.
  public override func forward(tensorBatch: TensorBatch, context: NetworkContext) -> TensorBatch {
    let blockOuts = innerBlockSequential.predict(batch: tensorBatch, context: context)
    
    return blockOuts
  }
  
  @discardableResult
  internal func applyGradientsToInnerBlock(gradients: (weights: Tensor, biases: Tensor), learningRate: Tensor.Scalar) -> GradientOffsets {
    var weightGradients: [Tensor] = []
    var biasGradients: [Tensor] = []
    
    var currentWeightOffset: Int = 0
    var currentBiasOffset: Int = 0
    
    for layer in innerBlockSequential.layers {
      guard layer.usesOptimizer else {
        // add placeholder for layers that don't need gradients
        weightGradients.append(Tensor())
        biasGradients.append(Tensor())
        continue
      }
      
      // this should be correct given how we calculate weight size at each layer
      let size = layer.weights.size
      let totalWeights = size.rows * size.columns * size.depth
      let indexOffset = currentWeightOffset
      
      let biasSize = layer.biases.size
      let totalBiases = biasSize.rows * biasSize.columns * biasSize.depth
      let indexBiasOffset = currentBiasOffset
      
      let wStorage = TensorStorage.create(count: totalWeights)
      wStorage.pointer.update(from: gradients.weights.storage.pointer + indexOffset, count: totalWeights)
      let layerWeights = Tensor(storage: wStorage, size: size)

      let bStorage = TensorStorage.create(count: totalBiases)
      bStorage.pointer.update(from: gradients.biases.storage.pointer + indexBiasOffset, count: totalBiases)
      let layerBiases = Tensor(storage: bStorage, size: biasSize)
      
      weightGradients.append(layerWeights)
      biasGradients.append(layerBiases)
      
      currentWeightOffset += totalWeights
      currentBiasOffset += totalBiases
    }
    
    innerBlockSequential.apply(gradients: .init(input: [],
                                                weights: weightGradients,
                                                biases: biasGradients),
                               learningRate: learningRate)
    
    return (weights: currentWeightOffset, biases: currentBiasOffset)
    
  }
  
}
