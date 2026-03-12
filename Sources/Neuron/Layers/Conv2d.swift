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
    
    let outStorage = conv(tensor)
    let outSize = TensorSize(rows: outputSize.rows, columns: outputSize.columns, depth: filterCount)
    let out = Tensor(storage: outStorage, size: outSize, context: tensorContext)
    
    out.setGraph(tensor)

    return super.forward(tensor: out, context: context)
  }
  
  internal func backward(_ input: Tensor, _ delta: Tensor) -> (input: Tensor, weight: Tensor, bias: Tensor) {
    let inputDepth = inputSize.depth
    let fRows = filterSize.rows
    let fCols = filterSize.columns
    let fSliceSize = fRows * fCols
    let deltaRows = delta.size.rows
    let deltaCols = delta.size.columns
    let deltaSliceSize = deltaRows * deltaCols

    // Build flipped-transposed kernel table using pointer APIs.
    // flippedKernels[f * filterCount + i] = flip180 of filters[i] depth slice f
    let flippedKernelStorage = TensorStorage.create(count: inputDepth * filterCount * fSliceSize)
    for i in 0..<filterCount {
      for f in 0..<inputDepth {
        let srcPtr = filters[i].storage.pointer + f * fSliceSize
        let dstPtr = flippedKernelStorage.pointer + (f * filterCount + i) * fSliceSize
        NumSwiftFlat.flip180(signal: srcPtr, result: dstPtr, rows: fRows, columns: fCols)
      }
    }

    // Compute stride-pad shape once (same for every filter delta slice)
    let spShape: (rows: Int, columns: Int)
    if strides.rows > 1 || strides.columns > 1 {
      spShape = NumSwiftFlat.stridePadShape(inputSize: (rows: deltaRows, columns: deltaCols), strides: strides)
    } else {
      spShape = (rows: deltaRows, columns: deltaCols)
    }

    let dRows = Double(inputSize.rows) - Double(spShape.rows)
    let dCols = Double(inputSize.columns) - Double(spShape.columns)
    let paddingTop    = Int(ceil(dRows / 2.0))
    let paddingBottom = Int(floor(dRows / 2.0))
    let paddingLeft   = Int(ceil(dCols / 2.0))
    let paddingRight  = Int(floor(dCols / 2.0))
    let numPadding = NumSwiftPadding(top: paddingTop, left: paddingLeft,
                                     right: paddingRight, bottom: paddingBottom)
    let newRows    = spShape.rows + paddingTop + paddingBottom
    let newColumns = spShape.columns + paddingLeft + paddingRight
    let paddedDeltaSize = newRows * newColumns

    // Input gradients: one TensorStorage per input depth channel, accumulated across filters
    let inputSliceSize = inputSize.rows * inputSize.columns
    let inputGradStorage = TensorStorage.create(count: inputSliceSize * inputDepth)

    // Weight gradients: filterCount * inputDepth slices of size fSliceSize
    let wGradStorage = TensorStorage.create(count: filterCount * inputDepth * fSliceSize)

    // Temp buffers reused across iterations
    let stridePaddedBuf = TensorStorage.create(count: spShape.rows * spShape.columns)
    let paddedDeltaBuf  = TensorStorage.create(count: paddedDeltaSize)
    let convResultBuf   = TensorStorage.create(count: inputSliceSize)

    for i in 0..<filterCount {
      let deltaSlicePtr = delta.storage.pointer + i * deltaSliceSize

      // Stride-pad the delta slice into stridePaddedBuf
      NumSwiftFlat.stridePad1D(signal: deltaSlicePtr,
                                result: stridePaddedBuf.pointer,
                                strides: strides,
                                signalSize: (rows: deltaRows, columns: deltaCols))

      // Zero-pad the stride-padded delta into paddedDeltaBuf
      NumSwiftFlat.zeroPad1D(signal: stridePaddedBuf.pointer,
                              result: paddedDeltaBuf.pointer,
                              padding: numPadding,
                              inputSize: spShape)

      // Accumulate input gradients: for each input depth f, convolve paddedDelta with flipped kernel
      for f in 0..<inputDepth {
        let kernelPtr = flippedKernelStorage.pointer + (f * filterCount + i) * fSliceSize
        let inputGradPtr = inputGradStorage.pointer + f * inputSliceSize

        self.device.conv2d(signal: paddedDeltaBuf.pointer,
                           filter: kernelPtr,
                           result: convResultBuf.pointer,
                           strides: (1, 1),
                           padding: .same,
                           filterSize: filterSize,
                           inputSize: (rows: newRows, columns: newColumns),
                           batchCount: 1)

        NumSwiftFlat.add(inputGradPtr, convResultBuf.pointer,
                          result: inputGradPtr, count: inputSliceSize)
      }

      // Weight gradients for filter i
      calculateFilterGradientsInto(input,
                                    deltaSlicePtr: deltaSlicePtr,
                                    deltaSize: (rows: deltaRows, columns: deltaCols),
                                    filterIndex: i,
                                    resultPtr: wGradStorage.pointer + i * inputDepth * fSliceSize)
    }

    let inputTensor = Tensor(storage: inputGradStorage, size: inputSize)
    inputTensor.label = "conv2d-input"

    let wSize = TensorSize(rows: fRows, columns: fCols, depth: filterCount * inputDepth)
    let weightsTensor = Tensor(storage: wGradStorage, size: wSize)
    weightsTensor.label = "conv2d-weight"

    // Bias gradients: sum each delta depth slice
    let biasStorage = TensorStorage.create(count: delta.size.depth)
    for d in 0..<delta.size.depth {
      let slicePtr = delta.storage.pointer + d * deltaSliceSize
      biasStorage[d] = NumSwiftFlat.sum(slicePtr, count: deltaSliceSize)
    }
    let biasesTensor = Tensor(storage: biasStorage, size: biases.size)
    biasesTensor.label = "conv2d-bias"

    precondition(biasesTensor.shape == biases.shape)

    return (inputTensor, weightsTensor, biasesTensor)
  }
  
  /// Computes weight gradients for a single filter and writes them directly into `resultPtr`.
  /// `resultPtr` must point to a buffer of size `inputSize.depth * filterSize.rows * filterSize.columns`.
  internal func calculateFilterGradientsInto(_ input: Tensor,
                                              deltaSlicePtr: TensorStorage.Pointer,
                                              deltaSize: (rows: Int, columns: Int),
                                              filterIndex: Int,
                                              resultPtr: TensorStorage.Pointer) {
    let fSliceSize = filterSize.rows * filterSize.columns
    let inputSliceSize = inputSize.rows * inputSize.columns

    let extraPadding = padding.extra(inputSize: (inputSize.rows, inputSize.columns),
                                     filterSize: filterSize,
                                     stride: strides)
    let numPadding = NumSwiftPadding(top: extraPadding.top,
                                     left: extraPadding.left,
                                     right: extraPadding.right,
                                     bottom: extraPadding.bottom)

    let paddedRows    = inputSize.rows + numPadding.top + numPadding.bottom
    let paddedColumns = inputSize.columns + numPadding.left + numPadding.right
    let paddedSize    = paddedRows * paddedColumns

    // Determine stride-padded filter shape once
    let filterInputSize: (rows: Int, columns: Int)
    if strides.rows > 1 || strides.columns > 1 {
      filterInputSize = NumSwiftFlat.stridePadShape(inputSize: deltaSize, strides: strides)
    } else {
      filterInputSize = deltaSize
    }
    let stridePaddedDeltaSize = filterInputSize.rows * filterInputSize.columns
    let convStrides = (filterSize.columns == 1 || filterSize.rows == 1) ? strides : (1, 1)

    let paddedSignalBuf      = TensorStorage.create(count: paddedSize)
    let stridePaddedDeltaBuf = TensorStorage.create(count: stridePaddedDeltaSize)

    for i in 0..<inputSize.depth {
      let signalPtr = input.storage.pointer + i * inputSliceSize

      // Zero-pad the input depth slice
      NumSwiftFlat.zeroPad1D(signal: signalPtr,
                              result: paddedSignalBuf.pointer,
                              padding: numPadding,
                              inputSize: (rows: inputSize.rows, columns: inputSize.columns))

      // Stride-pad the delta slice (used as the filter in weight grad conv)
      NumSwiftFlat.stridePad1D(signal: deltaSlicePtr,
                                result: stridePaddedDeltaBuf.pointer,
                                strides: strides,
                                signalSize: deltaSize)

      // Convolve: signal=paddedInput, filter=stridePaddedDelta → weight gradient slice
      self.device.conv2d(signal: paddedSignalBuf.pointer,
                         filter: stridePaddedDeltaBuf.pointer,
                         result: resultPtr + i * fSliceSize,
                         strides: convStrides,
                         padding: .valid,
                         filterSize: filterInputSize,
                         inputSize: (rows: paddedRows, columns: paddedColumns),
                         batchCount: 1)
    }
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
      let offset = startDepth * sliceSize
      let count = slicesPerFilter * sliceSize
      let gradStorage = TensorStorage.create(count: count)
      gradStorage.pointer.update(from: weightGradTensor.storage.pointer + offset, count: count)
      let gradTensor = Tensor(storage: gradStorage,
                              size: TensorSize(rows: filterSize.rows, columns: filterSize.columns, depth: slicesPerFilter))
      filters[i] = filters[i].copy() - gradTensor
    }

    if biasEnabled {
      biases = biases.copy() - gradients.biases
    }
  }
  
  // supports processing multiple in a batch at once
  internal func conv(_ input: Tensor) -> TensorStorage {
    let batchCount = inputSize.batchCount
    let outRows = outputSize.rows
    let outCols = outputSize.columns
    let outSliceSize = outRows * outCols * batchCount
    let inputSliceSize = inputSize.rows * inputSize.columns
    let filterSliceSize = filterSize.rows * filterSize.columns

    let resultStorage = TensorStorage.create(count: outSliceSize * filterCount)

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
          self.device.conv2d(signal: signalPtr,
                             filter: filterPtr,
                             result: resultPtr,
                             strides: strides,
                             padding: padding,
                             filterSize: filterSize,
                             inputSize: (inputSize.rows,
                                         inputSize.columns),
                             batchCount: batchCount)
        } else {
          self.device.conv2d(signal: signalPtr,
                             filter: filterPtr,
                             result: tempBuf.pointer,
                             strides: strides,
                             padding: padding,
                             filterSize: filterSize,
                             inputSize: (inputSize.rows,
                                         inputSize.columns),
                             batchCount: batchCount)
          
          NumSwiftFlat.add(resultPtr,
                           tempBuf.pointer,
                           result: resultPtr,
                           count: outSliceSize)
        }
      }

      if biasEnabled {
        let biasVal = biases.storage[f]
        NumSwiftFlat.add(resultPtr, scalar: biasVal, result: resultPtr, count: outSliceSize)
      }
    }

    return resultStorage
  }
  
}
