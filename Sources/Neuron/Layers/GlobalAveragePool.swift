//
//  GlobalAveragePool.swift
//
//

import Foundation
import NumSwift

/// A layer that computes the global average of each feature map, reducing spatial dimensions to a single value per channel.
public final class GlobalAvgPool: BaseLayer {
  /// Creates a global-average-pooling layer.
  ///
  /// - Parameter inputSize: Optional input shape; if supplied, output shape is derived immediately.
  public init(inputSize: TensorSize = TensorSize(array: []),
              linkId: String = UUID().uuidString) {
    super.init(inputSize: inputSize,
               biasEnabled: false,
               linkId: linkId,
               encodingType: .globalAvgPool)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize, type, linkId
  }
  
  override public func onInputSizeSet() {
    // outputSize will effectively 'flatten' with a global average at each channel
    outputSize = .init(rows: 1, columns: inputSize.depth, depth: 1)
  }
  
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    self.init(linkId: linkId)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
  /// Encodes global-average-pooling configuration.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(linkId, forKey: .linkId)
  }
  
  /// Computes global spatial means per channel.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Tensor of per-channel averages with shape `(1, depth, 1)`.
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    
    let tensorContext = TensorContext { [inputSize] inputs, gradient, wrt in
      // each input value at a given space only contributed 1 / (row * column) of the output
      let spatialCount = inputSize.rows * inputSize.columns
      let scale = Tensor.Scalar(1) / Tensor.Scalar(spatialCount)
      
      var outStorage = Tensor.Value(repeating: 0, count: spatialCount * inputSize.depth)
      
      for d in 0..<inputSize.depth {
        let gradientAtDepth = gradient.storage[d] * scale
        let depthOffset = d * spatialCount
        for i in 0..<spatialCount {
          outStorage[depthOffset + i] = gradientAtDepth
        }
      }
      
      return (Tensor(outStorage, size: inputSize), Tensor(), Tensor())
    }
    
    // Compute mean of each depth slice directly from flat storage
    let size = tensor.size
    let spatialCount = size.rows * size.columns
    let spatialScalar = Tensor.Scalar(spatialCount)
    var outValues = Tensor.Value(repeating: 0, count: size.depth)
    
    for d in 0..<size.depth {
      let depthOffset = d * spatialCount
      var sum: Tensor.Scalar = 0
      for i in 0..<spatialCount {
        sum += tensor.storage[depthOffset + i]
      }
      outValues[d] = sum / spatialScalar
    }

    // forward calculation - output is (depth, 1, 1)
    let outSize = TensorSize(rows: 1, columns: size.depth, depth: 1)
    let out = Tensor(outValues, size: outSize, context: tensorContext)
    
    out.setGraph(tensor)
    
    return super.forward(tensor: out, context: context)
  }
}

