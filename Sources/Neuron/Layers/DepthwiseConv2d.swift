//
//  DepthwiseConv2d.swift
//
//
//  Created by William Vabrinskas on 5/5/22.
//

import Foundation
import NumSwift
import NumSwiftC

/// A layer that performs a depthwise 2D convolution operation.
///
/// Depthwise convolution applies a single convolutional filter independently to each input channel
/// (depth slice), rather than mixing information across channels as a standard ``Conv2d`` does.
/// The number of output channels always equals the number of input channels.
///
/// This is the first step in a **depthwise separable convolution** (as used in MobileNet), where it
/// is typically followed by a pointwise (`Conv2d` with a 1×1 kernel) convolution that combines the
/// per-channel features.
///
/// ## How it differs from ``Conv2d``
/// | Property | `Conv2d` | `DepthwiseConv2d` |
/// |---|---|---|
/// | Filters per channel | `filterCount` shared across all channels | 1 dedicated filter per channel |
/// | Output depth | `filterCount` | same as input depth |
/// | Parameter count | `filterCount × inputDepth × kH × kW` | `inputDepth × kH × kW` |
///
/// ## Usage
/// ```swift
/// // Stand-alone depthwise conv on a 32×32 RGB image
/// let layer = DepthwiseConv2d(
///     inputSize: TensorSize(rows: 32, columns: 32, depth: 3),
///     strides: (1, 1),
///     padding: .same,
///     filterSize: (3, 3)
/// )
///
/// // Typical depthwise-separable block
/// let sequential = Sequential {[
///     DepthwiseConv2d(strides: (1, 1), padding: .same, filterSize: (3, 3)),
///     Conv2d(filterCount: 64, filterSize: (1, 1))   // pointwise step
/// ]}
/// ```
///
/// ## Output shape
/// Given an input of shape `(rows: H, columns: W, depth: D)` and filter size `(rows: kH, columns: kW)`:
/// - **`.same` padding**: output is `(rows: H/strideR, columns: W/strideC, depth: D)`
/// - **`.valid` padding**: output is `(rows: (H−kH)/strideR+1, columns: (W−kW)/strideC+1, depth: D)`
///
/// ## Gradient computation
/// During backpropagation each filter gradient and input gradient is computed independently per
/// channel using full convolution of the upstream gradient with the flipped filter kernel. This
/// keeps the backward pass embarrassingly parallel across channels.
public class DepthwiseConv2d: BaseConvolutionalLayer {
  /// Creates a depthwise 2D convolutional layer.
  ///
  /// Only the **first** layer in a network needs `inputSize` set explicitly; subsequent layers
  /// derive their input size automatically when the network is compiled by an ``Optimizer``.
  ///
  /// - Parameters:
  ///   - inputSize: Optional input size at this layer. Required only when this is the first layer.
  ///   - strides: Number of rows and columns to advance the filter window at each step. Default: `(1, 1)`
  ///   - padding: Padding strategy applied before convolution. `.same` preserves spatial dimensions
  ///     (with stride 1); `.valid` performs no padding. Default: `.valid`
  ///   - filterSize: Height and width of each convolutional kernel. Default: `(3, 3)`
  ///   - initializer: Strategy used to initialise filter weights. Default: `.heNormal`
  ///   - biasEnabled: When `true`, a scalar bias is added to every output element in each channel.
  ///     Default: `false`
  public init(inputSize: TensorSize? = nil,
              strides: (rows: Int, columns: Int) = (1,1),
              padding: NumSwift.ConvPadding = .valid,
              filterSize: (rows: Int, columns: Int) = (3,3),
              initializer: InitializerType = .heNormal,
              biasEnabled: Bool = false,
              linkId: String = UUID().uuidString,
              encodingType: EncodingType = .depthwiseConv2d) {
    
    super.init(filterCount: 1,
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
  
  /// Decodes a DepthwiseConv2d layer from a serialized model.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: An error if required values cannot be decoded.
  required convenience public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    let filterSize = try container.decodeIfPresent([Int].self, forKey: .filterSize) ?? [3,3]
    let strides = try container.decodeIfPresent([Int].self, forKey: .strides) ?? [1,1]
    let padding = try container.decodeIfPresent(NumSwift.ConvPadding.self, forKey: .padding) ?? .same
    let biasEnabled = try container.decodeIfPresent(Bool.self, forKey: .biasEnabled) ?? false
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    
    let filterSizeTuple = (filterSize[safe: 1, 3], filterSize[safe: 0, 3])
    let stridesTuple = (strides[safe: 1, 3], strides[safe: 0, 3])
    
    self.init(inputSize: inputSize,
              strides: stridesTuple,
              padding: padding,
              filterSize: filterSizeTuple,
              initializer: .heUniform,
              biasEnabled: biasEnabled,
              linkId: linkId)
    
    self.filters = try container.decodeIfPresent([Tensor].self, forKey: .filters) ?? []
  }
  
  /// Encodes the layer's configuration and parameters into the given encoder.
  ///
  /// - Parameter encoder: The encoder to write data to.
  /// - Throws: An error if any of the values fail to encode.
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
  
  /// Called automatically when `inputSize` is set.
  ///
  /// Computes `outputSize` using the standard convolution output formula (accounting for padding
  /// and strides), then initialises one bias scalar per input channel when `biasEnabled` is `true`.
  public override func onInputSizeSet() {
    super.onInputSizeSet()
    
    let paddingValue = padding.extra(inputSize: (self.inputSize.rows, self.inputSize.columns), filterSize: filterSize)
    
    let rows = (((self.inputSize.rows + (paddingValue.top + paddingValue.bottom)) - (filterSize.rows - 1) - 1) / strides.rows) + 1
    let columns = (((self.inputSize.columns + (paddingValue.left + paddingValue.right)) - (filterSize.columns - 1) - 1) / strides.columns) + 1
    
    outputSize = TensorSize(array: [columns, rows, inputSize.depth])
    
    if biasEnabled && biases.isEmpty {
      biases = Tensor([Tensor.Scalar](repeating: 0, count: inputSize.depth))
    }
  }
  
  /// Initialises the per-channel filter bank using the layer's ``initializer``.
  ///
  /// Creates one filter tensor of shape `(rows: filterSize.rows, columns: filterSize.columns, depth: 1)`
  /// for every input channel. The method is a no-op when filters have already been set (e.g. after
  /// decoding a saved model).
  override func initializeFilters() {
    guard filters.isEmpty else {
      return
    }
    
    // 1 filter per channel
    filters = (0..<inputSize.depth).map { _ in initializer.calculate(size: .init(rows: filterSize.rows,
                                                                                 columns: filterSize.columns,
                                                                                 depth: 1),
                                                                     input: inputSize.depth * filterSize.rows * filterSize.columns,
                                                                     out: inputSize.depth * filterSize.rows * filterSize.columns) }
  }
  
  /// Performs the depthwise forward convolution pass.
  ///
  /// For each input channel `i`, the corresponding filter `filters[i]` is convolved with that
  /// channel's depth slice. The results are concatenated depth-wise to form the output tensor.
  /// An optional per-channel bias is added when `biasEnabled` is `true`.
  ///
  /// A ``TensorContext`` is attached to the output so that the backward pass can be triggered
  /// automatically during gradient computation.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor of shape `(columns: W, rows: H, depth: D)`.
  ///   - context: Network context carrying batch metadata. Defaults to a default-constructed context.
  /// - Returns: Output tensor of shape `(columns: W', rows: H', depth: D)` where `W'` and `H'`
  ///   depend on the padding and stride settings.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    
    let tensorContext = TensorContext { inputs, gradient, wrt in
      self.backward(inputs, gradient)
    }
    
    let out = Tensor(storage: conv(tensor).storage, size: outputSize, context: tensorContext)
    
    out.setGraph(tensor)
    
    return super.forward(tensor: out, context: context)
  }
  
  /// Computes input, weight, and bias gradients for the depthwise convolution.
  ///
  /// Each channel is processed independently:
  /// 1. **Input gradient** – The upstream gradient slice for channel `i` is stride-padded (when
  ///    `strides > 1`) and zero-padded to restore spatial dimensions, then convolved with the
  ///    180°-flipped filter kernel for that channel.
  /// 2. **Weight gradient** – Computed by cross-correlating the input slice with the (optionally
  ///    stride-padded) upstream gradient slice, yielding a `(kH × kW)` gradient per channel.
  /// 3. **Bias gradient** – The element-wise sum of the upstream gradient for each channel.
  ///
  /// - Parameters:
  ///   - input: The original input tensor from the forward pass, shape `(W, H, D)`.
  ///   - delta: Upstream gradient tensor, shape `(W', H', D)`.
  /// - Returns: A tuple of `(input: Tensor, weight: Tensor, bias: Tensor)` gradients.
  ///   - `input`: Shape matches the forward-pass input, `(W, H, D)`.
  ///   - `weight`: Shape `(kW, kH, D)` – one gradient kernel per input channel.
  ///   - `bias`: Shape matches `biases`, one scalar per channel.
  internal func backward(_ input: Tensor, _ delta: Tensor) -> (input: Tensor, weight: Tensor, bias: Tensor) {
    let inputDepth = inputSize.depth
    let fRows = filterSize.rows
    let fCols = filterSize.columns
    let fSliceSize = fRows * fCols
    let deltaRows = delta.size.rows
    let deltaCols = delta.size.columns
    let deltaSliceSize = deltaRows * deltaCols
    let inputSliceSize = inputSize.rows * inputSize.columns

    // Pre-allocate one flipped-kernel storage block: inputDepth × fSliceSize
    let flippedKernelStorage = TensorStorage.create(count: inputDepth * fSliceSize)
    for i in 0..<inputDepth {
      let srcPtr = filters[i].storage.pointer   // depth=1, so slice 0 starts at base
      let dstPtr = flippedKernelStorage.pointer + i * fSliceSize
      NumSwiftFlat.flip180(signal: srcPtr, result: dstPtr, rows: fRows, columns: fCols)
    }

    // Compute stride-pad shape once (same for every channel)
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

    // Pre-allocate output storages
    let inputGradStorage  = TensorStorage.create(count: inputSliceSize * inputDepth)
    let weightGradStorage = TensorStorage.create(count: fSliceSize * inputDepth)

    // Temp buffers reused across iterations (one set per channel; backward is serial per channel)
    let stridePaddedBuf = TensorStorage.create(count: spShape.rows * spShape.columns)
    let paddedDeltaBuf  = TensorStorage.create(count: paddedDeltaSize)

    // Pre-allocate scratch buffers for calculateFilterGradientsInto (reused across inputDepth loop)
    let filterExtraPadding = padding.extra(inputSize: (inputSize.rows, inputSize.columns),
                                           filterSize: filterSize,
                                           stride: strides)
    let filterNumPadding = NumSwiftPadding(top: filterExtraPadding.top,
                                           left: filterExtraPadding.left,
                                           right: filterExtraPadding.right,
                                           bottom: filterExtraPadding.bottom)
    let filterPaddedRows    = inputSize.rows + filterNumPadding.top + filterNumPadding.bottom
    let filterPaddedColumns = inputSize.columns + filterNumPadding.left + filterNumPadding.right
    let filterPaddedSize    = filterPaddedRows * filterPaddedColumns
    let filterInputSize: (rows: Int, columns: Int) = (strides.rows > 1 || strides.columns > 1)
      ? NumSwiftFlat.stridePadShape(inputSize: (rows: deltaRows, columns: deltaCols), strides: strides)
      : (rows: deltaRows, columns: deltaCols)
    let stridePaddedDeltaCount = filterInputSize.rows * filterInputSize.columns
    let paddedSignalBuf      = TensorStorage.create(count: filterPaddedSize)
    let stridePaddedDeltaBuf = TensorStorage.create(count: stridePaddedDeltaCount)

    for i in 0..<inputDepth {
      let deltaSlicePtr  = delta.storage.pointer + i * deltaSliceSize
      let kernelPtr      = flippedKernelStorage.pointer + i * fSliceSize
      let inputGradPtr   = inputGradStorage.pointer + i * inputSliceSize
      let weightGradPtr  = weightGradStorage.pointer + i * fSliceSize

      // Stride-pad delta slice
      NumSwiftFlat.stridePad1D(signal: deltaSlicePtr,
                                result: stridePaddedBuf.pointer,
                                strides: strides,
                                signalSize: (rows: deltaRows, columns: deltaCols))

      // Zero-pad the stride-padded delta
      NumSwiftFlat.zeroPad1D(signal: stridePaddedBuf.pointer,
                              result: paddedDeltaBuf.pointer,
                              padding: numPadding,
                              inputSize: spShape)

      // Input gradient: convolve padded delta with flipped kernel
      device.conv2d(signal: paddedDeltaBuf.pointer,
                    filter: kernelPtr,
                    result: inputGradPtr,
                    strides: (1, 1),
                    padding: .same,
                    filterSize: filterSize,
                    inputSize: (rows: newRows, columns: newColumns),
                    batchCount: 1)

      // Weight gradient
      calculateFilterGradientsInto(inputSlicePtr: input.storage.pointer + i * inputSliceSize,
                                   deltaSlicePtr: deltaSlicePtr,
                                   deltaSize: (rows: deltaRows, columns: deltaCols),
                                   resultPtr: weightGradPtr,
                                   paddedSignalBuf: paddedSignalBuf,
                                   stridePaddedDeltaBuf: stridePaddedDeltaBuf,
                                   filterNumPadding: filterNumPadding,
                                   filterPaddedRows: filterPaddedRows,
                                   filterPaddedColumns: filterPaddedColumns,
                                   filterInputSize: filterInputSize)
    }

    let inputTensor = Tensor(storage: inputGradStorage, size: inputSize)
    inputTensor.label = "conv2d-input"

    let weightsTensor = Tensor(storage: weightGradStorage,
                               size: TensorSize(rows: fRows, columns: fCols, depth: inputDepth))
    weightsTensor.label = "conv2d-weight"

    // Bias gradients: sum each delta depth slice
    let biasStorage = TensorStorage.create(count: delta.size.depth)
    for d in 0..<delta.size.depth {
      biasStorage[d] = NumSwiftFlat.sum(delta.storage.pointer + d * deltaSliceSize, count: deltaSliceSize)
    }
    let biasesTensor = Tensor(storage: biasStorage, size: biases.size)
    biasesTensor.label = "conv2d-bias"

    precondition(biasesTensor.shape == biases.shape)

    return (inputTensor, weightsTensor, biasesTensor)
  }
  
  /// Computes the gradient with respect to a single filter kernel, writing directly into `resultPtr`.
  ///
  /// The caller must pre-allocate `paddedSignalBuf` and `stridePaddedDeltaBuf` and supply the
  /// derived padding/size values so these buffers are reused across the outer inputDepth loop.
  ///
  /// - Parameters:
  ///   - inputSlicePtr: Pointer to one input channel, length `inputSize.rows × inputSize.columns`.
  ///   - deltaSlicePtr: Pointer to the upstream gradient for the same channel.
  ///   - deltaSize: Spatial dimensions `(rows, columns)` of the delta slice.
  ///   - resultPtr: Destination pointer for the filter gradient, length `filterSize.rows × filterSize.columns`.
  ///   - paddedSignalBuf: Pre-allocated scratch buffer of size `filterPaddedRows × filterPaddedColumns`.
  ///   - stridePaddedDeltaBuf: Pre-allocated scratch buffer of size `filterInputSize.rows × filterInputSize.columns`.
  ///   - filterNumPadding: Forward-pass padding values (pre-computed by caller).
  ///   - filterPaddedRows: Padded input rows (pre-computed by caller).
  ///   - filterPaddedColumns: Padded input columns (pre-computed by caller).
  ///   - filterInputSize: Stride-padded delta shape (pre-computed by caller).
  internal func calculateFilterGradientsInto(inputSlicePtr: TensorStorage.Pointer,
                                             deltaSlicePtr: TensorStorage.Pointer,
                                             deltaSize: (rows: Int, columns: Int),
                                             resultPtr: TensorStorage.Pointer,
                                             paddedSignalBuf: TensorStorage,
                                             stridePaddedDeltaBuf: TensorStorage,
                                             filterNumPadding: NumSwiftPadding,
                                             filterPaddedRows: Int,
                                             filterPaddedColumns: Int,
                                             filterInputSize: (rows: Int, columns: Int)) {
    let convStrides = (filterSize.columns == 1 || filterSize.rows == 1) ? strides : (1, 1)

    NumSwiftFlat.zeroPad1D(signal: inputSlicePtr,
                            result: paddedSignalBuf.pointer,
                            padding: filterNumPadding,
                            inputSize: (rows: inputSize.rows, columns: inputSize.columns))

    NumSwiftFlat.stridePad1D(signal: deltaSlicePtr,
                              result: stridePaddedDeltaBuf.pointer,
                              strides: strides,
                              signalSize: deltaSize)

    device.conv2d(signal: paddedSignalBuf.pointer,
                  filter: stridePaddedDeltaBuf.pointer,
                  result: resultPtr,
                  strides: convStrides,
                  padding: .valid,
                  filterSize: filterInputSize,
                  inputSize: (rows: filterPaddedRows, columns: filterPaddedColumns),
                  batchCount: 1)
  }
  
  /// Applies pre-scaled weight and bias gradients to update the layer's filters.
  ///
  /// The optimizer calls this method after computing and scaling gradients. For each input channel
  /// `i`, the corresponding depth slice of `gradients.weights` is subtracted from `filters[i]`.
  /// When `biasEnabled` is `true`, `gradients.biases` is subtracted from `biases` element-wise.
  ///
  /// - Note: The learning-rate scaling is performed by the optimizer before calling this method,
  ///   so the raw gradient tensors already incorporate it.
  ///
  /// - Parameters:
  ///   - gradients: An ``Optimizer/Gradient`` value containing weight and bias gradient tensors
  ///     whose shapes match the layer's filters and biases respectively.
  ///   - learningRate: The current learning rate scalar (included for protocol conformance; the
  ///     optimizer typically pre-multiplies gradients before calling this method).
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    // in depthwise the gradient should be inputSize exactly
    
    let sliceSize = filterSize.rows * filterSize.columns
    for i in 0..<inputSize.depth {
      let gradStorage = TensorStorage.create(count: sliceSize)
      gradStorage.pointer.update(from: gradients.weights.storage.pointer + i * sliceSize, count: sliceSize)
      let gradTensor = Tensor(storage: gradStorage,
                              size: TensorSize(rows: filterSize.rows, columns: filterSize.columns, depth: 1))
      filters[i] = filters[i].copy() - gradTensor
    }
    
    if biasEnabled {
      biases = biases.copy() - gradients.biases
    }
  }
  
  /// Executes the depthwise convolution across all channels and returns flat output storage.
  ///
  /// Iterates over each input depth slice in parallel (up to `Constants.maxWorkers` threads),
  /// convolving it with its dedicated filter. Results are written into a pre-allocated flat array
  /// in depth-major order: all elements of channel 0 first, then channel 1, and so on.
  ///
  /// - Parameter input: Input tensor of shape `(W, H, D)`.
  /// - Returns: Flat output storage of length `outputSize.rows × outputSize.columns × inputSize.depth`.
  internal func conv(_ input: Tensor) -> Tensor {
    let outSliceSize = outputSize.rows * outputSize.columns
    let inputSliceSize = inputSize.rows * inputSize.columns
    let resultStorage = TensorStorage.create(count: outSliceSize * outputSize.depth)

    Array(0..<self.inputSize.depth).concurrentForEach(workers: Constants.maxWorkers) { _, i in
      let inputPtr  = input.storage.pointer + i * inputSliceSize
      let filterPtr = self.filters[i].storage.pointer   // depth=1, slice 0 starts at base
      let destPtr   = resultStorage.pointer + i * outSliceSize

      self.device.conv2d(signal: inputPtr,
                         filter: filterPtr,
                         result: destPtr,
                         strides: self.strides,
                         padding: self.padding,
                         filterSize: self.filterSize,
                         inputSize: (self.inputSize.rows, self.inputSize.columns),
                         batchCount: 1)
    }

    if biasEnabled {
      let biasT = Tensor(storage: biases.storage, size: TensorSize(rows: 1, columns: 1, depth: outputSize.depth))
      return Tensor(storage: resultStorage, size: outputSize) + biasT
    }

    return Tensor(storage: resultStorage, size: outputSize)
  }
  

}
