//
//  LayerGroup.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/25/26.
//

import NumSwift
import Foundation

public protocol LayerGrouping: Layer, BaseLayer {
  var innerBlockSequential: Sequential { get }
}


public class BaseLayerGroup: BaseLayer, LayerGrouping {
  
  public typealias GradientOffsets = (weights: Int, biases: Int)
  
  public var innerBlockSequential: Sequential = .init()
  
  private var layers: (_ inputSize: TensorSize) -> [Layer]
  
  public override var isTraining: Bool {
    get {
      super.isTraining
    }
    set {
      innerBlockSequential.isTraining = newValue
    }
  }

  public override var weights: Tensor {
    get {
      let innerBlockWeights = (try? innerBlockSequential.exportWeights()) ?? []
      var flatInnerBlockWeights: [Tensor] = innerBlockWeights.fullFlatten()
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
    // we detach the input tensor because we want to stop at this tensor in respect to this sequential
    // not all the way up the graph to possibly other input layers
    let blockOut = innerBlockSequential.predict(batch: [tensor], context: .init())[safe: 0, Tensor()]
    blockOut.label = "blockOut"
    
    return super.forward(tensor: blockOut, context: context)
  }
  
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
      
      let layerWeights = Tensor(Tensor.Value(gradients.weights.storage[indexOffset..<indexOffset + totalWeights]), size: size)
      let layerBiases = Tensor(Tensor.Value(gradients.biases.storage[indexBiasOffset..<indexBiasOffset + totalBiases]), size: biasSize)
      
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
