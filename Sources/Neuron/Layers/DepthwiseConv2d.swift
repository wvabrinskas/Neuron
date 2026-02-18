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
              encodingType: EncodingType = .depthwiseConv2d) {
    
    super.init(filterCount: 1,
               inputSize: inputSize,
               strides: strides,
               padding: padding,
               filterSize: filterSize,
               initializer: initializer,
               biasEnabled: biasEnabled,
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
         type
  }
  
  required convenience public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    let filterSize = try container.decodeIfPresent([Int].self, forKey: .filterSize) ?? [3,3]
    let strides = try container.decodeIfPresent([Int].self, forKey: .strides) ?? [1,1]
    let padding = try container.decodeIfPresent(NumSwift.ConvPadding.self, forKey: .padding) ?? .same
    let biasEnabled = try container.decodeIfPresent(Bool.self, forKey: .biasEnabled) ?? false
    
    let filterSizeTuple = (filterSize[safe: 1, 3], filterSize[safe: 0, 3])
    let stridesTuple = (strides[safe: 1, 3], strides[safe: 0, 3])
    
    self.init(inputSize: inputSize,
              strides: stridesTuple,
              padding: padding,
              filterSize: filterSizeTuple,
              initializer: .heUniform,
              biasEnabled: biasEnabled)
    
    self.filters = try container.decodeIfPresent([Tensor].self, forKey: .filters) ?? []
  }
  
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
    
    if biasEnabled {
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
    
    let context = TensorContext { inputs, gradient, wrt in
      self.backward(inputs, gradient)
    }
    
    let outStorage = conv(tensor)
    let out = Tensor(outStorage, size: outputSize, context: context)
    
    out.setGraph(tensor)
    out.label = "depthwiseConv2d"
    
    return out
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
    // Flip and transpose filters: for each input depth channel f, collect the flipped kernel[f] from each filter i
    // flippedTransposed[f][i] = flip180(filters[i].depthSlice(f))
    let inputDepth = inputSize.depth
    let fRows = filterSize.rows
    let fCols = filterSize.columns
    
    // Build flipped-transposed kernel table as flat arrays
    var flippedKernels = [Tensor.Value](repeating: Tensor.Value(), count: inputDepth)
    
    for i in 0..<inputDepth {
      let kernel = filters[i].storage
      flippedKernels[i] = NumSwiftFlat.flip180(kernel, rows: fRows, columns: fCols)
    }
    
    
    let deltaRows = delta.size.rows
    let deltaCols = delta.size.columns
    
    // Weight gradients: flat storage for filterCount * inputDepth depth slices
    let weightsTensor = Tensor(Tensor.Value(repeating: 0, count: filterSize.rows * filterSize.columns * inputDepth),
                               size: .init(rows: filterSize.rows,
                                           columns: filterSize.columns,
                                           depth: inputDepth))
    weightsTensor.label = "conv2d-weight"
    
    // Input gradients: one flat slice per input depth channel
    let inputTensor = Tensor(Tensor.Value(repeating: 0,
                                          count: inputSize.rows * inputSize.columns * inputDepth),
                             size: inputSize)
    inputTensor.label = "conv2d-input"
    
    var cachedStridePadShape: (rows: Int, columns: Int)?
    
    for i in 0..<inputDepth {
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
      
      let kernel = flippedKernels[i]
      
      let grad = device.conv2d(signal: workingDelta, filter: kernel,
                               strides: (1,1), padding: .same,
                               filterSize: filterSize,
                               inputSize: (rows: newRows, columns: newColumns),
                               outputSize: nil)
      
      inputTensor.setDepthSlice(i, grad)
      
      let filterGrads = calculateFilterGradientsFlat(input.depthSlice(i),
                                                     deltaSlice,
                                                     deltaSize: (rows: deltaRows, columns: deltaCols),
                                                     index: i)
      
      weightsTensor.setDepthSlice(i, filterGrads)
    }
    
    
    // Assemble weight gradients tensor
    let biasStorage = Tensor.Value((0..<delta.size.depth).map { delta.depthSlice($0).sum })
    let biasesTensor = Tensor(biasStorage, size: biases.size)
    biasesTensor.label = "conv2d-bias"
    
    precondition(biasesTensor.shape == biases.shape)
    
    /// --------- input gradients end ----------
    
    return (inputTensor, weightsTensor, biasesTensor)
  }
  
  /// Computes the gradient with respect to a single filter kernel.
  ///
  /// Implements the weight-gradient step of backpropagation for one input channel by
  /// cross-correlating the (padded) input slice with the (stride-inserted) upstream gradient.
  ///
  /// - Parameters:
  ///   - input: Flat storage for one input channel, length `inputSize.rows × inputSize.columns`.
  ///   - delta: Flat upstream gradient for the same channel, length `deltaSize.rows × deltaSize.columns`.
  ///   - deltaSize: Spatial dimensions `(rows, columns)` of `delta` before any stride insertion.
  ///   - index: Channel index used only for logging / labelling purposes.
  /// - Returns: Flat filter gradient of length `filterSize.rows × filterSize.columns`.
  internal func calculateFilterGradientsFlat(_ input: Tensor.Value,
                                             _ delta: Tensor.Value,
                                             deltaSize: (rows: Int, columns: Int),
                                             index: Int) -> Tensor.Value {
    
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
    
    var filter = delta
    var signal = input
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
    
    return result
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
    
    for i in 0..<inputSize.depth {
      // Extract this filter's gradient slices and build a tensor
      let gradient = gradients.weights.depthSlice(i)
      
      let gradTensor = Tensor(gradient,
                              size: TensorSize(rows: filterSize.rows,
                                               columns: filterSize.columns,
                                               depth: 1))
      
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
  internal func conv(_ input: Tensor) -> Tensor.Value {
    var resultStorage: Tensor = Tensor(Tensor.Value(repeating: 0, count: outputSize.columns * outputSize.rows * outputSize.depth), size: outputSize)
    
    Array(0..<self.inputSize.depth).concurrentForEach(workers: Constants.maxWorkers) { _, i in
      let currentFilter = self.filters[i].storage
      let currentInput = input.depthSlice(i)
      
      var conv = self.device.conv2d(signal: currentInput,
                                    filter: currentFilter,
                                    strides: self.strides,
                                    padding: self.padding,
                                    filterSize: self.filterSize,
                                    inputSize: (self.inputSize.rows, self.inputSize.columns),
                                    outputSize: nil)
      
      if self.biasEnabled {
        let bias = self.biases.storage[i]
        conv = conv + bias
      }
      
      resultStorage.setDepthSlice(i, conv)
    }
    
    return resultStorage.storage
  }
  
  /// Returns a 180°-rotated copy of every depth slice in the given filter tensor.
  ///
  /// Used during backpropagation to flip kernels before the full convolution that computes
  /// input gradients. Each depth slice is rotated independently.
  ///
  /// - Parameter filter: Filter tensor with arbitrary depth; each slice has shape
  ///   `(filter.size.rows × filter.size.columns)`.
  /// - Returns: An array of flat depth slices, each 180°-rotated, with the same length as the
  ///   filter's depth.
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
