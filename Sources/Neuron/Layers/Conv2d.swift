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
public class Conv2d<N: TensorNumeric>: BaseConvolutionalLayer<N> {
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
                       encodingType: EncodingType = .conv2d) {
    
    super.init(filterCount: filterCount,
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
    let filterCount = try container.decodeIfPresent(Int.self, forKey: .filterCount) ?? 1
    let strides = try container.decodeIfPresent([Int].self, forKey: .strides) ?? [1,1]
    let padding = try container.decodeIfPresent(NumSwift.ConvPadding.self, forKey: .padding) ?? .same
    let biasEnabled = try container.decodeIfPresent(Bool.self, forKey: .biasEnabled) ?? false
    
    let filterSizeTuple = (filterSize[safe: 1, 3], filterSize[safe: 0, 3])
    let stridesTuple = (strides[safe: 1, 3], strides[safe: 0, 3])
    
    self.init(filterCount: filterCount,
              inputSize: inputSize,
              strides: stridesTuple,
              padding: padding,
              filterSize: filterSizeTuple,
              initializer: .heUniform,
              biasEnabled: biasEnabled)
    
    self.filters = try container.decodeIfPresent([Tensor<N>].self, forKey: .filters) ?? []
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
  
  public override func onInputSizeSet() {
    super.onInputSizeSet()
    
    let paddingValue = padding.extra(inputSize: (self.inputSize.rows, self.inputSize.columns), filterSize: filterSize)
    
    let rows = (((self.inputSize.rows + (paddingValue.top + paddingValue.bottom)) - (filterSize.rows - 1) - 1) / strides.rows) + 1
    let columns = (((self.inputSize.columns + (paddingValue.left + paddingValue.right)) - (filterSize.columns - 1) - 1) / strides.columns) + 1
    
    outputSize = TensorSize(array: [columns, rows, filterCount])
  }
  
  public override func forward(tensor: Tensor<N>) -> Tensor<N> {
    let context = TensorContext<N> { inputs, gradient in
      self.backward(inputs, gradient)
    }
    
    let out = Tensor<N>(conv(tensor), context: context)
    
    out.setGraph(tensor)

    return out
  }
  
  internal func backward(_ input: Tensor<N>, _ delta: Tensor<N>) -> (input: Tensor<N>, weight: Tensor<N>, bias: Tensor<N>) {
    let flippedTransposed = filters.map { flip180($0) }.transposed() as [[[[Tensor<N>.Scalar]]]]
    
    var weightGradients: [[[Tensor<N>.Scalar]]] = []
    var inputGradients: [[[Tensor<N>.Scalar]]] = []

    var cachedDeltaShape: [Int]?
    
    for i in 0..<filterCount {
      let delta = delta.value[i]
      var workingDelta = NumSwiftC.stridePad(signal: delta, strides: strides)
      
      let deltaShape: [Int]
      if let cachedDeltaShape {
        deltaShape = cachedDeltaShape
      } else {
        deltaShape = workingDelta.shape
        cachedDeltaShape = deltaShape
      }
      
      let deltaRows = Double(inputSize.rows) - Double(deltaShape[safe: 1, 0])
      let deltaColumns = Double(inputSize.columns) - Double(deltaShape[safe: 0, 0])
      
      let paddingTop = Int(ceil(deltaRows / Double(2)))
      let paddingBottom = Int(floor(deltaRows / Double(2)))
      let paddingLeft = Int(ceil(deltaColumns / Double(2)))
      let paddingRight = Int(floor(deltaColumns / Double(2)))
      
      let numPadding = NumSwiftPadding(top: paddingTop,
                                       left: paddingLeft,
                                       right: paddingRight,
                                       bottom: paddingBottom)
      
      workingDelta = NumSwiftC.zeroPad(signal: workingDelta, padding: numPadding)
      
      let newRows = deltaShape[safe: 1, 0] + paddingTop + paddingBottom
      let newColumns = deltaShape[safe: 0, 0] + paddingLeft + paddingRight
      
      for f in 0..<flippedTransposed.count {
        let filter = flippedTransposed[f]
        let kernel = filter[i]
        
        let gradientsForKernelIndex: [[Tensor<N>.Scalar]] = device.conv2d(signal: workingDelta,
                                                                       filter: kernel,
                                                                       strides: (1,1),
                                                                       padding: .same,
                                                                       filterSize: filterSize,
                                                                       inputSize: (newRows, newColumns),
                                                                       outputSize: nil)
        
        if let currentGradientsForFilter = inputGradients[safe: f] {
          inputGradients[f] = currentGradientsForFilter + gradientsForKernelIndex
        } else {
          inputGradients.append(gradientsForKernelIndex)
        }
      }
      
      let filterGradients = calculateFilterGradients(input, delta, index: i)
      weightGradients.append(contentsOf: filterGradients)
    }
    
    let biasGradients = delta.value.map { $0.sum }
    
    return (Tensor<N>(inputGradients), Tensor<N>(weightGradients), Tensor<N>(biasGradients))
  }
  
  internal func calculateFilterGradients(_ input: Tensor<N>, _ delta: [[Tensor<N>.Scalar]], index: Int) -> Tensor<N>.Data {
    var newGradientsForFilters: Tensor<N>.Data = []
    var cachedFilterShape: [Int]?

    
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
      var signal = input.value[i]
      
      signal = NumSwiftC.zeroPad(signal: signal, padding: numPadding)
      filter = NumSwiftC.stridePad(signal: filter, strides: strides)
    
      let fShape: [Int]
      if let cachedFilterShape {
        fShape = cachedFilterShape
      } else {
        fShape = filter.shape
        cachedFilterShape = fShape
      }
      
      //TODO: figure out valid with strides. need to pad right and bottom for filter
      //
      //
      //      if padding == .valid {
      //        let filterPadding = NumSwift.ConvPadding.same.extra(inputSize: filterSize,
      //                                                            filterSize: (sShape[safe: 1, 0], sShape[safe: 0, 0]),
      //                                                            stride: (1,1))
      //
      //        filter = filter.zeroPad(padding: NumSwiftPadding(top: filterPadding.top - (filterSize.rows - 1),
      //                                                         left: filterPadding.left - (filterSize.rows - 1),
      //                                                         right: filterPadding.right - (filterSize.rows - 1),
      //                                                         bottom: filterPadding.bottom - (filterSize.rows - 1)))
      //        fShape = filter.shape
      //      }
      
      let newFilterSize = (fShape[safe: 1] ?? 0, fShape[safe: 0] ?? 0)
      
      let result = device.conv2d(signal: signal,
                                 filter: filter,
                                 strides: (1,1),
                                 padding: .valid,
                                 filterSize: newFilterSize,
                                 inputSize: convInputSize,
                                 outputSize: nil)
      
      newGradientsForFilters.append(result)
    }
    //all filter gradients will be mashed into one 3D array and then batched out later by num of filters
    //this way we dont have to store these gradients
    return newGradientsForFilters
  }
  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor<N>.Scalar) {
      
    //batch out gradients by number of filters
    var weightGradientsBatched = [gradients.weights.value]
    
    if filterCount > 1 {
      weightGradientsBatched = gradients.weights.value.batched(into: inputSize.depth)
    }
    
    for i in 0..<filterCount {
      let filterGradients = weightGradientsBatched[i]
      filters[i] = filters[i] - Tensor<N>(filterGradients)
    }
    
    if biasEnabled {
      biases = biases - gradients.biases
    }
  }
  
  internal func conv(_ input: Tensor<N>) -> [[[Tensor<N>.Scalar]]] {
    var results: [[[Tensor<N>.Scalar]]] = []
    
    let flatBias: [Tensor<N>.Scalar] = biases.value.flatten()
    
    for f in 0..<filterCount {
      var convolved: [[Tensor<N>.Scalar]] = [] // maybe do concurrentForEach here too

      for i in 0..<inputSize.depth {
        let currentFilter = self.filters[f].value[i]
        let currentInput = input.value[i]
        
        let conv = self.device.conv2d(signal: currentInput,
                                      filter: currentFilter,
                                      strides: self.strides,
                                      padding: self.padding,
                                      filterSize: self.filterSize,
                                      inputSize: (self.inputSize.rows, self.inputSize.columns),
                                      outputSize: nil)
        
        if convolved.isEmpty {
          convolved = conv
        } else {
          convolved = convolved + conv
        }
      }
      
      if self.biasEnabled {
        let bias = flatBias[f]
        convolved = convolved + bias
      }
      
      results.append(convolved)
    }

    return results
  }
  
  private func flip180(_ filter: Tensor<N>) -> [[[Tensor<N>.Scalar]]] {
    filter.value.map { $0.flip180() }
  }
}
