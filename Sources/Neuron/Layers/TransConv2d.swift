//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/5/22.
//

import Foundation
import NumSwift
import NumSwiftC

/// Performs a transposed 2d convolution on the inputs. Uses the same properties and initializers of `Conv2D`
public final class TransConv2d: Conv2d {
  public override init(filterCount: Int,
                       inputSize: TensorSize? = nil,
                       strides: (rows: Int, columns: Int) = (1,1),
                       padding: NumSwift.ConvPadding = .valid,
                       filterSize: (rows: Int, columns: Int) = (3,3),
                       initializer: InitializerType = .heNormal,
                       biasEnabled: Bool = false,
                       encodingType: EncodingType = .transConv2d) {
    
    super.init(filterCount: filterCount,
               inputSize: inputSize,
               strides: strides,
               padding: padding,
               filterSize: filterSize,
               initializer: initializer,
               biasEnabled: biasEnabled,
               encodingType: .transConv2d)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    
    var rows = self.inputSize.rows * strides.rows
    var columns = self.inputSize.columns * strides.columns
    
    if padding == .valid {
      rows = (self.inputSize.rows - 1) * strides.rows + filterSize.rows
      columns = (self.inputSize.columns - 1) * strides.columns + filterSize.columns
    }
    
    outputSize = TensorSize(array: [columns, rows, filterCount])
  }
  
  internal override func backward(_ input: Tensor, _ delta: Tensor) -> (input: Tensor, weight: Tensor, bias: Tensor) {
    let inputDepth = inputSize.depth
    let fRows = filterSize.rows
    let fCols = filterSize.columns
    let deltaRows = delta.size.rows
    let deltaCols = delta.size.columns
    
    // Build flipped-transposed kernel table as flat arrays
    var flippedKernels = [Tensor.Value](repeating: Tensor.Value(), count: inputDepth * filterCount)
    for i in 0..<filterCount {
      for f in 0..<inputDepth {
        let kernel = filters[i].depthSlice(f)
        flippedKernels[f * filterCount + i] = NumSwiftFlat.flip180(kernel, rows: fRows, columns: fCols)
      }
    }
    
    var weightGradientSlices = [Tensor.Value]()
    var inputGradientSlices = [Tensor.Value?](repeating: nil, count: inputDepth)
    
    for i in 0..<filterCount {
      let deltaSlice = delta.depthSlice(i)
      
      for f in 0..<inputDepth {
        let kernel = flippedKernels[f * filterCount + i]
        
        let grad = NumSwiftFlat.conv2d(signal: deltaSlice, filter: kernel,
                                       strides: strides, padding: padding,
                                       filterSize: filterSize,
                                       inputSize: (rows: deltaRows, columns: deltaCols))
        
        if let existing = inputGradientSlices[f] {
          inputGradientSlices[f] = NumSwiftFlat.add(existing, grad)
        } else {
          inputGradientSlices[f] = grad
        }
      }
      
      if trainable {
        let filterGrads = calculateFilterGradientsFlat(input, deltaSlice,
                                                        deltaSize: (rows: deltaRows, columns: deltaCols),
                                                        index: i)
        // Insert at beginning (matching original behavior)
        weightGradientSlices.insert(contentsOf: filterGrads, at: 0)
      }
    }
    
    // Bias gradients: sum of each depth slice of input
    var biasStorage = Tensor.Value(repeating: 0, count: inputDepth)
    for d in 0..<inputDepth {
      biasStorage[d] = NumSwiftFlat.sum(input.depthSlice(d))
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
    
    // Assemble weight gradients tensor
    let wSliceSize = fRows * fCols
    var wStorage = Tensor.Value(repeating: 0, count: wSliceSize * weightGradientSlices.count)
    for (idx, slice) in weightGradientSlices.enumerated() {
      let start = idx * wSliceSize
      for j in 0..<min(slice.count, wSliceSize) {
        wStorage[start + j] = slice[j]
      }
    }
    
    return (Tensor(inputStorage, size: inputSize),
            Tensor(wStorage, size: TensorSize(rows: fRows, columns: fCols, depth: weightGradientSlices.count)),
            Tensor(biasStorage, size: TensorSize(rows: 1, columns: inputDepth, depth: 1)))
  }
  
  internal override func calculateFilterGradientsFlat(_ input: Tensor,
                                              _ delta: Tensor.Value,
                                              deltaSize: (rows: Int, columns: Int),
                                              index: Int) -> [Tensor.Value] {
    var results = [Tensor.Value]()
    results.reserveCapacity(inputSize.depth)
    
    var cachedSignalSize: (rows: Int, columns: Int)?
    
    for i in 0..<inputSize.depth {
      var filter = input.depthSlice(i)
      var signal = delta
      var filterInputSize = (rows: inputSize.rows, columns: inputSize.columns)
      
      if strides.0 > 1 {
        let spShape = NumSwiftFlat.stridePadShape(inputSize: filterInputSize, strides: strides)
        filter = NumSwiftFlat.stridePad(signal: filter, strides: strides, inputSize: filterInputSize)
        filterInputSize = spShape
        // Add extra padding on right and bottom
        filter = NumSwiftFlat.zeroPad(signal: filter,
                                       padding: NumSwiftPadding(top: 0, left: 0,
                                                                right: strides.0 - 1,
                                                                bottom: strides.1 - 1),
                                       inputSize: filterInputSize)
        filterInputSize = (rows: filterInputSize.rows + strides.1 - 1,
                          columns: filterInputSize.columns + strides.0 - 1)
      }
      
      var signalSize = deltaSize
      if padding == .same {
        signal = NumSwiftFlat.zeroPad(signal: signal, filterSize: filterSize,
                                       inputSize: (rows: outputSize.rows, columns: outputSize.columns))
        let padCalc = NumSwiftFlat.paddingCalculation(strides: (1,1), padding: .same,
                                                       filterSize: filterSize,
                                                       inputSize: (rows: outputSize.rows, columns: outputSize.columns))
        signalSize = (rows: outputSize.rows + padCalc.top + padCalc.bottom,
                     columns: outputSize.columns + padCalc.left + padCalc.right)
      }
      
      if cachedSignalSize == nil {
        cachedSignalSize = signalSize
      } else {
        signalSize = cachedSignalSize!
      }
      
      let newFilterSize = (filterInputSize.rows, filterInputSize.columns)
      
      let result = NumSwiftFlat.conv2d(signal: signal, filter: filter,
                                       strides: (1,1), padding: .valid,
                                       filterSize: newFilterSize,
                                       inputSize: signalSize)
      
      results.append(result)
    }
    
    return results
  }
  
  internal override func conv(_ input: Tensor) -> Tensor.Value {
    let outRows = outputSize.rows
    let outCols = outputSize.columns
    let outSliceSize = outRows * outCols
    let inRows = inputSize.rows
    let inCols = inputSize.columns
    
    // Initialize result storage for all filter outputs
    var filterOutputs = [Tensor.Value?](repeating: nil, count: filterCount)
    
    for i in 0..<input.depthSliceCount {
      let localInput = input.depthSlice(i)
      var workingInput = NumSwiftFlat.stridePad(signal: localInput, strides: strides,
                                                 inputSize: (rows: inRows, columns: inCols))
      
      let spShape = (strides.rows > 1 || strides.columns > 1)
        ? NumSwiftFlat.stridePadShape(inputSize: (rows: inRows, columns: inCols), strides: strides)
        : (rows: inRows, columns: inCols)
      
      let inputRowsDiff = Double(outRows) - Double(spShape.rows)
      let inputColsDiff = Double(outCols) - Double(spShape.columns)
      
      let paddingTop = Int(ceil(inputRowsDiff / Double(2)))
      let paddingBottom = Int(floor(inputRowsDiff / Double(2)))
      let paddingLeft = Int(ceil(inputColsDiff / Double(2)))
      let paddingRight = Int(floor(inputColsDiff / Double(2)))
      
      let numPadding = NumSwiftPadding(top: paddingTop, left: paddingLeft,
                                       right: paddingRight, bottom: paddingBottom)
      
      workingInput = NumSwiftFlat.zeroPad(signal: workingInput, padding: numPadding, inputSize: spShape)
      
      let newRows = spShape.rows + paddingTop + paddingBottom
      let newColumns = spShape.columns + paddingLeft + paddingRight
      
      for f in 0..<filterCount {
        let kernel = filters[f].depthSlice(i)
        
        var grad = NumSwiftFlat.conv2d(signal: workingInput, filter: kernel,
                                       strides: (1,1), padding: .same,
                                       filterSize: filterSize,
                                       inputSize: (rows: newRows, columns: newColumns))
        
        if let existing = filterOutputs[f] {
          grad = NumSwiftFlat.add(grad, existing)
        }
        
        if biasEnabled {
          grad = NumSwiftFlat.add(grad, scalar: biases.storage[f])
        }
        
        filterOutputs[f] = grad
      }
    }
    
    // Assemble flat result
    var resultStorage = Tensor.Value(repeating: 0, count: outSliceSize * filterCount)
    for f in 0..<filterCount {
      if let output = filterOutputs[f] {
        let start = f * outSliceSize
        for j in 0..<min(output.count, outSliceSize) {
          resultStorage[start + j] = output[j]
        }
      }
    }
    
    return resultStorage
  }
  
}
