//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/5/22.
//

import Foundation
import NumSwift
import NumSwiftC

/// A layer that performs a 2D convolution operation
public class Conv2d: BaseConvolutionalLayer {
  /// Default initializer for a 2d convolutional layer
  /// - Parameters:
  ///   - filterCount: Number of filters at this layer
  ///   - inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  ///   - strides: Number of row and column strides when performing convolution. Default: `(3,3)`
  ///   - padding: Padding type when performing the convolution. Default: `.valid`
  ///   - filterSize: Size of the filter kernel. Default: `(3,3)`
  ///   - initializer: Weight / filter initializer function. Default: `.heNormal`
  ///   - biasEnabled: Boolean defining if the filters have a bias applied. Default: `false`
  public override init(filterCount: Int,
                       inputSize: TensorSize? = nil,
                       strides: (rows: Int, columns: Int) = (1,1),
                       padding: NumSwift.ConvPadding = .valid,
                       filterSize: (rows: Int, columns: Int) = (3,3),
                       initializer: InitializerType = .heNormal,
                       biasEnabled: Bool = false,
                       linkId: String = UUID().uuidString,
                       encodingType: EncodingType = .conv2d) {
    
    super.init(filterCount: filterCount,
               inputSize: inputSize,
               strides: strides,
               padding: padding,
               filterSize: filterSize,
               initializer: initializer,
               biasEnabled: biasEnabled,
               linkId: linkId,
               encodingType: encodingType)
  }
  
  enum CodingKeys: String, CodingKey {
    case filterSize,
         filterCount,
         strides,
         filters,
         padding,
         inputSize,
         biasEnabled,
         type,
         linkId
  }
  
  /// Decodes a convolution layer and restores its learned parameters.
  ///
  /// - Parameter decoder: Decoder containing serialized layer state.
  /// - Throws: Decoding errors when payload is invalid.
  required convenience public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    let filterSize = try container.decodeIfPresent([Int].self, forKey: .filterSize) ?? [3,3]
    let filterCount = try container.decodeIfPresent(Int.self, forKey: .filterCount) ?? 1
    let strides = try container.decodeIfPresent([Int].self, forKey: .strides) ?? [1,1]
    let padding = try container.decodeIfPresent(NumSwift.ConvPadding.self, forKey: .padding) ?? .same
    let biasEnabled = try container.decodeIfPresent(Bool.self, forKey: .biasEnabled) ?? false
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    
    let filterSizeTuple = (filterSize[safe: 1, 3], filterSize[safe: 0, 3])
    let stridesTuple = (strides[safe: 1, 3], strides[safe: 0, 3])
    
    self.init(filterCount: filterCount,
              inputSize: inputSize,
              strides: stridesTuple,
              padding: padding,
              filterSize: filterSizeTuple,
              initializer: .heUniform,
              biasEnabled: biasEnabled,
              linkId: linkId)
    
    self.filters = try container.decodeIfPresent([Tensor].self, forKey: .filters) ?? []
  }
  
  /// Encodes convolution configuration and filter parameters.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode([filterSize.rows, filterSize.columns], forKey: .filterSize)
    try container.encode([strides.rows, strides.columns], forKey: .strides)
    try container.encode(padding, forKey: .padding)
    try container.encode(biasEnabled, forKey: .biasEnabled)
    try container.encode(filterCount, forKey: .filterCount)
    try container.encode(filters, forKey: .filters)
    try container.encode(encodingType, forKey: .type)
    try container.encode(linkId, forKey: .linkId)
  }
  
  /// Recomputes output shape when input shape changes.
  public override func onInputSizeSet() {
    super.onInputSizeSet()
    
    let paddingValue = padding.extra(inputSize: (self.inputSize.rows, self.inputSize.columns), filterSize: filterSize)
    
    let rows = (((self.inputSize.rows + (paddingValue.top + paddingValue.bottom)) - (filterSize.rows - 1) - 1) / strides.rows) + 1
    let columns = (((self.inputSize.columns + (paddingValue.left + paddingValue.right)) - (filterSize.columns - 1) - 1) / strides.columns) + 1
    
    outputSize = TensorSize(array: [columns, rows, filterCount])
  }
  
  /// Performs a convolution forward pass and constructs backprop context.
  ///
  /// - Parameters:
  ///   - tensor: Input feature tensor.
  ///   - context: Network execution context.
  /// - Returns: Convolved output tensor.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {

    let tensorContext = TensorContext { inputs, gradient, wrt in
      self.backward(inputs, gradient)
    }
    
    let outStorage = conv(tensor, context: context)
    let outSize = TensorSize(rows: outputSize.rows, columns: outputSize.columns, depth: filterCount)
    let out = Tensor(storage: outStorage, size: outSize, context: tensorContext)
    
    out.setGraph(tensor)

    return super.forward(tensor: out, context: context)
  }
  
  internal func backward(_ input: Tensor, _ delta: Tensor) -> (input: Tensor, weight: Tensor, bias: Tensor) {
    // Flip and transpose filters: for each input depth channel f, collect the flipped kernel[f] from each filter i
    // flippedTransposed[f][i] = flip180(filters[i].depthSlice(f))
    let inputDepth = inputSize.depth
    let fRows = filterSize.rows
    let fCols = filterSize.columns
    
    // Build flipped-transposed kernel table as flat arrays
    // flippedKernels[f * filterCount + i] = flip180 of filters[i] depth slice f
    var flippedKernels = [Tensor.Value](repeating: Tensor.Value(), count: inputDepth * filterCount)
    for i in 0..<filterCount {
      for f in 0..<inputDepth {
        let kernel = filters[i].depthSlice(f)
        flippedKernels[f * filterCount + i] = NumSwiftFlat.flip180(kernel, rows: fRows, columns: fCols)
      }
    }
    
    let deltaRows = delta.size.rows
    let deltaCols = delta.size.columns
    
    // Weight gradients: flat storage for filterCount * inputDepth depth slices
    var weightGradientSlices = [Tensor.Value]()
    weightGradientSlices.reserveCapacity(filterCount * inputDepth)
    
    // Input gradients: one flat slice per input depth channel
    var inputGradientSlices = [Tensor.Value?](repeating: nil, count: inputDepth)
    
    var cachedStridePadShape: (rows: Int, columns: Int)?
    
    for i in 0..<filterCount {
      let deltaSlice = delta.depthSlice(i)
      var workingDelta = NumSwiftFlat.stridePad(signal: deltaSlice, strides: strides,
                                                 inputSize: (rows: deltaRows, columns: deltaCols))
      
      let spShape: (rows: Int, columns: Int)
      if let cachedStridePadShape {
        spShape = cachedStridePadShape
      } else {
        if strides.rows > 1 || strides.columns > 1 {
          spShape = NumSwiftFlat.stridePadShape(inputSize: (rows: deltaRows, columns: deltaCols), strides: strides)
        } else {
          spShape = (rows: deltaRows, columns: deltaCols)
        }
        cachedStridePadShape = spShape
      }
      
      let dRows = Double(inputSize.rows) - Double(spShape.rows)
      let dCols = Double(inputSize.columns) - Double(spShape.columns)
      
      let paddingTop = Int(ceil(dRows / Double(2)))
      let paddingBottom = Int(floor(dRows / Double(2)))
      let paddingLeft = Int(ceil(dCols / Double(2)))
      let paddingRight = Int(floor(dCols / Double(2)))
      
      let numPadding = NumSwiftPadding(top: paddingTop, left: paddingLeft,
                                       right: paddingRight, bottom: paddingBottom)
      
      workingDelta = NumSwiftFlat.zeroPad(signal: workingDelta, padding: numPadding,
                                           inputSize: spShape)
      
      let newRows = spShape.rows + paddingTop + paddingBottom
      let newColumns = spShape.columns + paddingLeft + paddingRight
      
      for f in 0..<inputDepth {
        let kernel = flippedKernels[f * filterCount + i]
        
        let grad = device.conv2d(signal: workingDelta, filter: kernel,
                                 strides: (1,1), padding: .same,
                                 filterSize: filterSize,
                                 inputSize: (rows: newRows, columns: newColumns),
                                 outputSize: nil)
        
        if let existing = inputGradientSlices[f] {
          inputGradientSlices[f] = existing + grad
        } else {
          inputGradientSlices[f] = grad
        }
      }
      
      let filterGrads = calculateFilterGradientsFlat(input, deltaSlice,
                                                      deltaSize: (rows: deltaRows, columns: deltaCols),
                                                      index: i)
      weightGradientSlices.append(contentsOf: filterGrads)
    }
    
    // Assemble input gradients tensor
    let inputSliceSize = inputSize.rows * inputSize.columns
    var inputStorage = Tensor.Value(repeating: 0, count: inputSliceSize * inputDepth)
    for f in 0..<inputDepth {
      if let slice = inputGradientSlices[f] {
        let start = f * inputSliceSize
        for j in 0..<min(slice.count, inputSliceSize) {
          inputStorage[start + j] = slice[j]
        }
      }
    }
    
    let inputTensor = Tensor(inputStorage, size: inputSize)
    inputTensor.label = "conv2d-input"
    
    // Assemble weight gradients tensor
    let wSliceSize = fRows * fCols
    var wStorage = Tensor.Value(repeating: 0, count: wSliceSize * weightGradientSlices.count)
    for (idx, slice) in weightGradientSlices.enumerated() {
      let start = idx * wSliceSize
      for j in 0..<min(slice.count, wSliceSize) {
        wStorage[start + j] = slice[j]
      }
    }
    let wSize = TensorSize(rows: fRows, columns: fCols, depth: weightGradientSlices.count)
    let weightsTensor = Tensor(wStorage, size: wSize)
    weightsTensor.label = "conv2d-weight"
    
    let biasStorage = Tensor.Value((0..<delta.size.depth).map { delta.depthSlice($0).sum })
    let biasesTensor = Tensor(biasStorage, size: biases.size)
    biasesTensor.label = "conv2d-bias"
    
    precondition(biasesTensor.shape == biases.shape)
    
    return (inputTensor, weightsTensor, biasesTensor)
  }
  
  internal func calculateFilterGradientsFlat(_ input: Tensor,
                                              _ delta: Tensor.Value,
                                              deltaSize: (rows: Int, columns: Int),
                                              index: Int) -> [Tensor.Value] {
    var results = [Tensor.Value]()
    results.reserveCapacity(inputSize.depth)
    
    let extraPadding = padding.extra(inputSize: (inputSize.rows, inputSize.columns),
                                     filterSize: filterSize,
                                     stride: strides)
    
    let numPadding = NumSwiftPadding(top: extraPadding.top,
                                     left: extraPadding.left,
                                     right: extraPadding.right,
                                     bottom: extraPadding.bottom)
    
    let expectedRows = inputSize.rows + numPadding.top + numPadding.bottom
    let expectedColumns = inputSize.columns + numPadding.left + numPadding.right
    let convInputSize = (expectedRows, expectedColumns)

    for i in 0..<inputSize.depth {
      var filter = delta
      var signal = input.depthSlice(i)
      var filterInputSize = deltaSize
      
      signal = NumSwiftFlat.zeroPad(signal: signal, padding: numPadding,
                                     inputSize: (rows: inputSize.rows, columns: inputSize.columns))
      
      if strides.rows > 1 || strides.columns > 1 {
        let newShape = NumSwiftFlat.stridePadShape(inputSize: filterInputSize, strides: strides)
        filter = NumSwiftFlat.stridePad(signal: filter, strides: strides, inputSize: filterInputSize)
        filterInputSize = newShape
      }
      
      let newFilterSize = (filterInputSize.rows, filterInputSize.columns)
      
      let convStrides = (filterSize.columns == 1 || filterSize.rows == 1) ? strides : (1, 1)
      
      let result = device.conv2d(signal: signal, filter: filter,
                                 strides: convStrides, padding: .valid,
                                 filterSize: newFilterSize,
                                 inputSize: convInputSize,
                                 outputSize: nil)
      
      results.append(result)
    }
    
    return results
  }
  
  /// Applies convolution weight and bias updates from optimizer gradients.
  ///
  /// - Parameters:
  ///   - gradients: Weight and bias gradients for this layer.
  ///   - learningRate: Learning rate (already applied by optimizer gradient formulation).
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    // Split weight gradients into per-filter chunks (each filter has inputSize.depth depth slices)
    // effectively a 4D tensor where each filter is (filterRows x filterColumns x inputDepth) x filterCount or [rows, columns, depth, count]
    let weightGradTensor = gradients.weights
    let slicesPerFilter = inputSize.depth
    let sliceSize = filterSize.rows * filterSize.columns
    
    for i in 0..<filterCount {
      // Extract this filter's gradient slices and build a tensor
      let startDepth = i * slicesPerFilter
      let gradStorage = Tensor.Value(
        weightGradTensor.storage[(startDepth * sliceSize)..<((startDepth + slicesPerFilter) * sliceSize)]
      )
      let gradTensor = Tensor(gradStorage,
                               size: TensorSize(rows: filterSize.rows, columns: filterSize.columns, depth: slicesPerFilter))
      filters[i] = filters[i].copy() - gradTensor
    }

    if biasEnabled {
      biases = biases.copy() - gradients.biases
    }
  }
  
  internal func conv(_ input: Tensor, context: NetworkContext = .init()) -> TensorStorage {
    let outRows = outputSize.rows
    let outCols = outputSize.columns
    let outSliceSize = outRows * outCols
    let inputSliceSize = inputSize.rows * inputSize.columns
    let filterSliceSize = filterSize.rows * filterSize.columns

    let resultStorage = TensorStorage.create(count: outSliceSize * filterCount)

    if device is GPU,
       let metalInput = input.storage as? MetalTensorStorage,
       MetalContext.shared.isAvailable,
       let metalDevice = MetalContext.shared.device,
       let pool = MetalContext.shared.bufferPool {
      let engine = MetalEngine()
      let extraPadding = padding.extra(inputSize: (inputSize.rows, inputSize.columns),
                                       filterSize: filterSize,
                                       stride: strides)
      let N: UInt32 = 1
      let C = UInt32(inputSize.depth)
      let H = UInt32(inputSize.rows)
      let W = UInt32(inputSize.columns)
      let K = UInt32(filterCount)
      let kH = UInt32(filterSize.rows)
      let kW = UInt32(filterSize.columns)
      let oH = UInt32(outRows)
      let oW = UInt32(outCols)
      let strideH = UInt32(strides.rows)
      let strideW = UInt32(strides.columns)
      let padH = UInt32(extraPadding.top)
      let padW = UInt32(extraPadding.left)

      let weightsCount = Int(K) * Int(C) * Int(kH) * Int(kW)
      let weightsStorage = MetalTensorStorage(device: metalDevice, count: weightsCount, pool: pool)
      let filterSliceCount = Int(C) * Int(kH) * Int(kW)
      for f in 0..<filterCount {
        weightsStorage.pointer.advanced(by: f * filterSliceCount)
          .update(from: filters[f].storage.pointer, count: filterSliceCount)
      }

      let outputStorage = MetalTensorStorage(device: metalDevice, count: outSliceSize * filterCount, pool: pool)
      var biasStorage: MetalTensorStorage?
      if biasEnabled {
        let bias = MetalTensorStorage(device: metalDevice, count: filterCount, pool: pool)
        for f in 0..<filterCount {
          bias.pointer.advanced(by: f).initialize(to: biases.storage[f])
        }
        biasStorage = bias
      }

      let params = MetalEngine.Conv2DParams(
        N: N, C: C, H: H, W: W, K: K,
        kH: kH, kW: kW, oH: oH, oW: oW,
        strideH: strideH, strideW: strideW,
        padH: padH, padW: padW,
        hasBias: biasEnabled ? 1 : 0
      )

      if let enc = context.metalEncoder {
        if engine.encodeConv2d(
          encoder: enc,
          input: metalInput,
          weights: weightsStorage,
          output: outputStorage,
          bias: biasStorage,
          params: params
        ) {
          return outputStorage
        }
      } else if engine.dispatchConv2d(
        input: metalInput,
        weights: weightsStorage,
        output: outputStorage,
        bias: biasStorage,
        params: params
      ) {
        return outputStorage
      }
    }

    if device is CPU {
      let strides = self.strides
      let padding = self.padding
      let filterSize = self.filterSize
      let inputSize = self.inputSize
      let biasEnabled = self.biasEnabled
      let filters = self.filters
      let biases = self.biases

      Array(0..<filterCount).concurrentForEach(workers: Constants.maxWorkers) { _, f in
        let resultPtr = resultStorage.pointer + f * outSliceSize
        let tempBuf = TensorStorage.create(count: outSliceSize)

        for i in 0..<inputSize.depth {
          let signalPtr = input.storage.pointer + i * inputSliceSize
          let filterPtr = filters[f].storage.pointer + i * filterSliceSize

          if i == 0 {
            NumSwiftFlat.conv2d(signal: signalPtr, filter: filterPtr, result: resultPtr,
                               strides: strides, padding: padding,
                               filterSize: filterSize, inputSize: (inputSize.rows, inputSize.columns))
          } else {
            NumSwiftFlat.conv2d(signal: signalPtr, filter: filterPtr, result: tempBuf.pointer,
                               strides: strides, padding: padding,
                               filterSize: filterSize, inputSize: (inputSize.rows, inputSize.columns))
            NumSwiftFlat.add(resultPtr, tempBuf.pointer, result: resultPtr, count: outSliceSize)
          }
        }

        if biasEnabled {
          let biasVal = biases.storage[f]
          NumSwiftFlat.add(resultPtr, scalar: biasVal, result: resultPtr, count: outSliceSize)
        }
      }
    } else {
      var resultArray = Tensor.Value(repeating: 0, count: outSliceSize * filterCount)
      Array(0..<filterCount).concurrentForEach(workers: Constants.maxWorkers) { _, f in
        var convolved = Tensor.Value(repeating: 0, count: outSliceSize)
        for i in 0..<self.inputSize.depth {
          let currentFilter = self.filters[f].depthSlice(i)
          let currentInput = input.depthSlice(i)
          let conv = self.device.conv2d(signal: currentInput, filter: currentFilter,
                                       strides: self.strides, padding: self.padding,
                                       filterSize: self.filterSize,
                                       inputSize: (self.inputSize.rows, self.inputSize.columns),
                                       outputSize: nil)
          if conv.count == convolved.count {
            convolved = convolved + conv
          } else {
            for j in 0..<min(conv.count, convolved.count) {
              convolved[j] += conv[j]
            }
          }
        }
        if self.biasEnabled {
          convolved = convolved + self.biases.storage[f]
        }
        let start = f * outSliceSize
        for j in 0..<outSliceSize {
          resultArray[start + j] = convolved[j]
        }
      }
      return TensorStorage.create(from: resultArray)
    }

    return resultStorage
  }
  
  internal func flip180Flat(_ filter: Tensor) -> [Tensor.Value] {
    var result = [Tensor.Value]()
    result.reserveCapacity(filter.size.depth)
    let fRows = filter.size.rows
    let fCols = filter.size.columns
    for d in 0..<filter.size.depth {
      result.append(NumSwiftFlat.flip180(filter.depthSlice(d), rows: fRows, columns: fCols))
    }
    return result
  }
}
