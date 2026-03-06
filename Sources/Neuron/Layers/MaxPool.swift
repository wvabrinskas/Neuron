//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/5/22.
//

import Foundation
import NumSwift

/// Will decrease the size of the input tensor by half using a max pooling technique.
public final class MaxPool: BaseLayer {
  internal struct PoolingIndex: Hashable, Codable {
    var r: Int
    var c: Int
  }
  
  internal struct PoolingGradient: Hashable, Codable {
    static func == (lhs: MaxPool.PoolingGradient, rhs: MaxPool.PoolingGradient) -> Bool {
      lhs.tensorId == rhs.tensorId
    }
    
    var tensorId: Tensor.ID
    var indicies: [[PoolingIndex]]
  }
    
  /// Default initializer for max pooling.
  /// - Parameter inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(inputSize: TensorSize? = nil,
              linkId: String = UUID().uuidString) {
    super.init(inputSize: inputSize,
               biasEnabled: false,
               linkId: linkId,
               encodingType: .maxPool)
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         type,
         linkId
  }
  
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    self.init(linkId: linkId)
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
  /// Encodes max-pooling layer configuration.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(linkId, forKey: .linkId)
  }
  
  /// Batched Metal path: pack, single MaxPool dispatch, unpack. Falls back to CPU with sync when needed.
  public override func forward(tensorBatch: TensorBatch, context: NetworkContext) -> TensorBatch {
    let batchCount = tensorBatch.count
    guard batchCount > 1,
          device is GPU,
          context.metalEncoder != nil,
          MetalContext.shared.isAvailable,
          let metalDevice = MetalContext.shared.device,
          let pool = MetalContext.shared.bufferPool,
          !tensorBatch.isEmpty else {
      return forwardCPUSyncIfNeeded(tensorBatch: tensorBatch, context: context)
    }

    guard let packedInput = BatchLayout.packToNCHW(tensorBatch, device: metalDevice, pool: pool) else {
      return forwardCPUSyncIfNeeded(tensorBatch: tensorBatch, context: context)
    }

    let rows = inputSize.rows
    let columns = inputSize.columns
    let outRows = (rows + 1) / 2
    let outCols = (columns + 1) / 2
    let depth = inputSize.depth

    let N = UInt32(batchCount)
    let C = UInt32(depth)
    let H = UInt32(rows)
    let W = UInt32(columns)
    let outH = UInt32(outRows)
    let outW = UInt32(outCols)

    let outputCount = batchCount * outRows * outCols * depth
    let outputStorage = MetalTensorStorage(device: metalDevice, count: outputCount, pool: pool)
    let indicesStorage = MetalTensorStorage(device: metalDevice, count: outputCount, pool: pool)

    let engine = MetalEngine()
    guard engine.encodeMaxPool2x2(
      encoder: context.metalEncoder!,
      input: packedInput,
      output: outputStorage,
      indices: indicesStorage,
      N: N, C: C, H: H, W: W,
      outH: outH, outW: outW
    ) else {
      return forwardCPUSyncIfNeeded(tensorBatch: tensorBatch, context: context)
    }

    context.metalEncoderHolder?.syncAndReplace()

    let singleOutSize = TensorSize(rows: outRows, columns: outCols, depth: depth)
    let elementsPerSample = outRows * outCols * depth

    var outputs: [Tensor] = []
    outputs.reserveCapacity(batchCount)

    for sampleIdx in 0..<batchCount {
      let idxOffset = sampleIdx * elementsPerSample
      let tensorContext = TensorContext { [weak self, indicesStorage, idxOffset] inputs, gradient, wrt in
        guard let self else { return (Tensor(), Tensor(), Tensor()) }
        var outStorage = Tensor.Value(repeating: 0, count: self.inputSize.rows * self.inputSize.columns * self.inputSize.depth)
        let inRows = self.inputSize.rows
        let inCols = self.inputSize.columns

        indicesStorage.pointer.withMemoryRebound(to: UInt32.self, capacity: indicesStorage.count) { idxPtr in
          let basePtr = idxPtr.advanced(by: idxOffset)
          for d in 0..<self.inputSize.depth {
            let gradSlice = gradient.depthSlice(d)
            var deltaIdx = 0
            let depthOffset = d * inRows * inCols

            for oh in 0..<outRows {
              for ow in 0..<outCols {
                guard deltaIdx < gradSlice.count else { break }
                let gid = d * outRows * outCols + oh * outCols + ow
                let offset = basePtr[gid]
                let ih = oh * 2 + Int(offset == 1 || offset == 3 ? 1 : 0)
                let iw = ow * 2 + Int(offset == 2 || offset == 3 ? 1 : 0)
                outStorage[depthOffset + ih * inCols + iw] = gradSlice[deltaIdx]
                deltaIdx += 1
              }
            }
          }
        }

        return (Tensor(outStorage, size: self.inputSize), Tensor(), Tensor())
      }

      let slice = Tensor.Value(unsafeUninitializedCapacity: elementsPerSample) { buf, count in
        buf.baseAddress!.initialize(from: outputStorage.pointer.advanced(by: sampleIdx * elementsPerSample), count: elementsPerSample)
        count = elementsPerSample
      }
      let sliceStorage = MetalTensorStorage(device: metalDevice, data: slice, pool: pool)
      let tensor = Tensor(storage: sliceStorage, size: singleOutSize, context: tensorContext)
      tensor.setGraph(tensorBatch[sampleIdx])
      outputs.append(tensor)
    }

    return outputs
  }

  /// Syncs GPU then runs CPU forward. Used when Metal batched path is not taken.
  private func forwardCPUSyncIfNeeded(tensorBatch: TensorBatch, context: NetworkContext) -> TensorBatch {
    if let first = tensorBatch.first,
       first.storage is MetalTensorStorage,
       let holder = context.metalEncoderHolder {
      holder.syncAndReplace()
    }
    return super.forward(tensorBatch: tensorBatch, context: context)
  }

  /// Performs 2x2 max pooling on each depth slice.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Pooled tensor with routing indices captured for backpropagation.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    var poolingGradients: PoolingGradient = .init(tensorId: tensor.id, indicies: [])
    
    func backwards(input: Tensor, gradient: Tensor, wrt: Tensor?) -> (Tensor, Tensor, Tensor) {
      var outStorage = Tensor.Value(repeating: 0, count: self.inputSize.rows * self.inputSize.columns * self.inputSize.depth)
    
      // operation is performed first then returned
      
      let forwardPooledMaxIndicies = poolingGradients.indicies
      
      let inRows = inputSize.rows
      let inCols = inputSize.columns
      
      for d in 0..<inputSize.depth {
        let gradSlice = gradient.depthSlice(d)
        var deltaIdx = 0
        let indicies = forwardPooledMaxIndicies[d]
        let depthOffset = d * inRows * inCols
        
        for index in indicies {
          if deltaIdx < gradSlice.count {
            outStorage[depthOffset + index.r * inCols + index.c] = gradSlice[deltaIdx]
            deltaIdx += 1
          }
        }
      }

     // print(outStorage.count, inputSize.columns * inputSize.rows * inputSize.depth)
      return (Tensor(outStorage, size: self.inputSize), Tensor(), Tensor())
    }
    
    let rows = inputSize.rows
    let columns = inputSize.columns
    let outRows = (rows + 1) / 2
    let outCols = (columns + 1) / 2
    
    var currentIndicies: [[PoolingIndex]] = []
    currentIndicies.reserveCapacity(inputSize.depth)
    
    var outStorage = Tensor.Value(repeating: 0, count: outRows * outCols * inputSize.depth)

    for d in 0..<inputSize.depth {
      let slice = tensor.depthSlice(d)
      let (poolResult, indices) = poolFlat(input: slice, rows: rows, columns: columns)
      currentIndicies.append(indices)
      
      let depthOffset = d * outRows * outCols
      for j in 0..<poolResult.count {
        outStorage[depthOffset + j] = poolResult[j]
      }
    }

    poolingGradients = PoolingGradient(tensorId: tensor.id, indicies: currentIndicies)

    let tensorContext = TensorContext(backpropagate: backwards)
    let outSize = TensorSize(rows: outRows, columns: outCols, depth: inputSize.depth)
    let out = Tensor(outStorage, size: outSize, context: tensorContext)
    
    out.setGraph(tensor)
    
    return super.forward(tensor: out, context: context)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = TensorSize(array: [(inputSize.columns + 1) / 2, (inputSize.rows + 1) / 2, inputSize.depth])
  }
  
  private func setGradients(indicies: [[PoolingIndex]], id: Tensor.ID) {
  }
  
  /// MaxPool has no trainable parameters, so this is a no-op.
  ///
  /// - Parameters:
  ///   - gradients: Ignored.
  ///   - learningRate: Ignored.
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
  }
  
  internal func poolFlat(input: Tensor.Value, rows: Int, columns: Int) -> (Tensor.Value, [PoolingIndex]) {
    var results = Tensor.Value()
    var pooledIndicies: [PoolingIndex] = []
    
    func safeGet(_ r: Int, _ c: Int) -> Tensor.Scalar {
      guard r >= 0, r < rows, c >= 0, c < columns else { return 0 }
      return input[r * columns + c]
    }
    
    for r in stride(from: 0, to: rows, by: 2) {
      for c in stride(from: 0, to: columns, by: 2) {
        let current = safeGet(r, c)
        let right = safeGet(r + 1, c)
        let bottom = safeGet(r, c + 1)
        let diag = safeGet(r + 1, c + 1)
        
        let indiciesToCheck = [(current, r, c),
                               (right, r + 1, c),
                               (bottom, r, c + 1),
                               (diag, r + 1, c + 1)]
        
        let maxVal = Swift.max(Swift.max(Swift.max(current, right), bottom), diag)
        if let firstIndicies = indiciesToCheck.first(where: { $0.0 == maxVal }) {
          pooledIndicies.append(PoolingIndex(r: firstIndicies.1, c: firstIndicies.2))
        }
        results.append(maxVal)
      }
    }
    
    return (results, pooledIndicies)
  }
}
