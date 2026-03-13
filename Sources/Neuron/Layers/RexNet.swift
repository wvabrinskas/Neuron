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

/// A residual-style network block that applies an expansion ratio and optional skip connections,
/// building on `BaseLayerGroup` to support efficient feature extraction.
public final class RexNet: BaseLayerGroup {
  private let expandRatio: Tensor.Scalar
  private let strides: (rows: Int, columns: Int)
  private let outChannels: Int
  private let squeeze: Int
  
  private var shouldSkip: Bool {
    strides.columns == 1 && strides.rows == 1 && outChannels == inputSize.depth
  }
  
/// Creates a new `RexNet` layer group with the specified configuration.
/// - Parameter inputSize: The optional input tensor size for the layer. Defaults to `nil`.
/// - Parameter initializer: The weight initializer type to use. Defaults to the framework's default initializer.
/// - Parameter strides: The row and column strides for the convolution operation. Defaults to `(1, 1)`.
/// - Parameter outChannels: The number of output channels produced by this block.
/// - Parameter squeeze: The squeeze factor used in channel reduction. Defaults to `0`.
/// - Parameter expandRatio: The ratio by which channels are expanded before the depthwise convolution. Defaults to `0`.
/// - Parameter linkId: A unique string identifier used to link this layer. Defaults to a new UUID string.
  public init(inputSize: TensorSize? = nil,
              initializer: InitializerType = Constants.defaultInitializer,
              strides: (rows: Int, columns: Int) = (1,1),
              outChannels: Int,
              squeeze: Int = 0,
              expandRatio: Tensor.Scalar = 0,
              linkId: String = UUID().uuidString) {
    
    self.expandRatio = expandRatio
    self.strides = strides
    self.outChannels = outChannels
    self.squeeze = squeeze
    
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: false,
               linkId: linkId,
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
        Swish(linkId: "squeeze")
      ]
      
      layers.append(contentsOf: depthWiseLayers)
      
      // Step 3: Squeeze-and-Excite (optional)
      if squeeze > 0 {
        let squeezeLayers: [Layer] = [
          GlobalAvgPool(),
          Dense(Int(inputSize.depth) / squeeze,
                initializer: initializer,
                biasEnabled: true),
          ReLu(),
          Dense(inputSize.depth,
                initializer: initializer,
                biasEnabled: true),
          Sigmoid(),
          Multiply(linkTo: "squeeze")
        ]
        
        layers.append(contentsOf: squeezeLayers)
      }
      
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
    case inputSize, type, linkId, expandRatio, stridesRows, stridesColumns, outChannels, squeeze, innerBlockSequential
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    /// do something when the input size is set when calling `compile` on `Sequential`
    /// like setting the output size or initializing the weights
    outputSize = innerBlockSequential.layers.last?.outputSize ?? inputSize
  }
  
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    let expandRatio = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .expandRatio) ?? 0
    let stridesRows = try container.decodeIfPresent(Int.self, forKey: .stridesRows) ?? 1
    let stridesColumns = try container.decodeIfPresent(Int.self, forKey: .stridesColumns) ?? 1
    let outChannels = try container.decodeIfPresent(Int.self, forKey: .outChannels) ?? 1
    let squeeze = try container.decodeIfPresent(Int.self, forKey: .squeeze) ?? 0

    self.init(strides: (stridesRows, stridesColumns),
              outChannels: outChannels,
              squeeze: squeeze,
              expandRatio: expandRatio,
              linkId: linkId)

    let innerBlockSequential = try container.decodeIfPresent(Sequential.self, forKey: .innerBlockSequential) ?? Sequential()
    self.innerBlockSequential = innerBlockSequential

    // set inputSize AFTER restoring innerBlockSequential so onInputSizeSet() sees the decoded layers
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
/// Encodes this `RexNet` instance into the given encoder, storing the input size, encoding type, and link ID.
/// - Parameter encoder: The encoder to write data into.
/// - Throws: An error if any values fail to encode.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(linkId, forKey: .linkId)
    try container.encode(expandRatio, forKey: .expandRatio)
    try container.encode(strides.rows, forKey: .stridesRows)
    try container.encode(strides.columns, forKey: .stridesColumns)
    try container.encode(outChannels, forKey: .outChannels)
    try container.encode(squeeze, forKey: .squeeze)
    try container.encode(innerBlockSequential, forKey: .innerBlockSequential)
  }
  
/// Performs a forward pass over a batch of tensors, processing each tensor individually and collecting the results.
/// - Parameter tensorBatch: The batch of input tensors to process.
/// - Parameter context: The network context providing execution state and mode information.
/// - Returns: A `TensorBatch` containing the forward-pass output for each input tensor.
  public override func forward(tensorBatch: TensorBatch, context: NetworkContext) -> TensorBatch {
    var result: TensorBatch = []
    
    for tensor in tensorBatch {
      result.append(forward(tensor: tensor, context: context))
    }
    
    return result
  }
  
/// Performs a forward pass on a single tensor, applying a skip connection when the stride and channel conditions allow,
/// and registers a backpropagation context for gradient computation.
/// - Parameter tensor: The input tensor to process.
/// - Parameter context: The network context providing execution state and mode information.
/// - Returns: The output tensor after applying the block's transformation and optional residual addition.
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    
    let forwardPass = if shouldSkip {
      super.forward(tensor: tensor, context: context) + tensor
    } else {
      super.forward(tensor: tensor, context: context)
    }
    
    let tensorContext = TensorContext { inputs, gradient, wrt in
      // backpropogation calculation
      
      let mainFlowInputGradients = forwardPass.gradients(delta: gradient, wrt: inputs)
      
      let wrtInput = mainFlowInputGradients.input[safe: 0, Tensor()]
      
      // we can't just use the Add autograd because we need to pass all of the gradients in one giant tensor back
      let weightSlices = mainFlowInputGradients.weights
      let totalWeightCount = weightSlices.reduce(0) { $0 + $1.storage.count }
      let wGradStorage = TensorStorage.create(count: totalWeightCount)
      var wOffset = 0
      for t in weightSlices {
        let c = t.storage.count
        wGradStorage.pointer.advanced(by: wOffset).update(from: t.storage.pointer, count: c)
        wOffset += c
      }

      let biasSlices = mainFlowInputGradients.biases
      let totalBiasCount = biasSlices.reduce(0) { $0 + $1.storage.count }
      let bGradStorage = TensorStorage.create(count: totalBiasCount)
      var bOffset = 0
      for t in biasSlices {
        let c = t.storage.count
        bGradStorage.pointer.advanced(by: bOffset).update(from: t.storage.pointer, count: c)
        bOffset += c
      }

      let wrtWeights = Tensor(storage: wGradStorage, size: TensorSize(rows: 1, columns: totalWeightCount, depth: 1))
      let wrtBiases  = Tensor(storage: bGradStorage, size: TensorSize(rows: 1, columns: totalBiasCount, depth: 1))

      return (wrtInput,
              wrtWeights,
              wrtBiases)
    }
    
    // forward calculation
    let out = Tensor(storage: forwardPass.storage,
                     size: forwardPass.size,
                     context: tensorContext)
    
    out.setGraph(tensor)
    
    return out
  }
}
