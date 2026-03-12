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
  /// Creates a transposed-convolution layer.
  ///
  /// Parameters match `Conv2d` and define filter, stride, padding, and
  /// initializer behavior for transpose convolution.
  public override init(filterCount: Int,
                       inputSize: TensorSize? = nil,
                       strides: (rows: Int, columns: Int) = (1,1),
                       padding: NumSwift.ConvPadding = .valid,
                       filterSize: (rows: Int, columns: Int) = (3,3),
                       initializer: InitializerType = .heNormal,
                       biasEnabled: Bool = false,
                       linkId: String = UUID().uuidString,
                       encodingType: EncodingType = .transConv2d) {
    
    super.init(filterCount: filterCount,
               inputSize: inputSize,
               strides: strides,
               padding: padding,
               filterSize: filterSize,
               initializer: initializer,
               biasEnabled: biasEnabled,
               linkId: linkId,
               encodingType: .transConv2d)
  }
  
  /// Recomputes transposed-convolution output shape for the current input size.
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
    let fSliceSize = fRows * fCols
    let deltaRows = delta.size.rows
    let deltaCols = delta.size.columns
    let deltaSliceSize = deltaRows * deltaCols
    let inputSliceSize = inputSize.rows * inputSize.columns

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

    // Input gradients accumulated across all filters
    let inputGradStorage = TensorStorage.create(count: inputSliceSize * inputDepth)
    // Weight gradients: filterCount * inputDepth slices of size fSliceSize
    let wGradStorage = TensorStorage.create(count: filterCount * inputDepth * fSliceSize)

    let convResultBuf = TensorStorage.create(count: inputSliceSize)

    for i in 0..<filterCount {
      let deltaSlicePtr = delta.storage.pointer + i * deltaSliceSize

      for f in 0..<inputDepth {
        let kernelPtr = flippedKernelStorage.pointer + (f * filterCount + i) * fSliceSize
        let inputGradPtr = inputGradStorage.pointer + f * inputSliceSize

        NumSwiftFlat.conv2d(signal: deltaSlicePtr,
                             filter: kernelPtr,
                             result: convResultBuf.pointer,
                             strides: strides,
                             padding: padding,
                             filterSize: filterSize,
                             inputSize: (rows: deltaRows, columns: deltaCols))

        NumSwiftFlat.add(inputGradPtr, convResultBuf.pointer,
                          result: inputGradPtr, count: inputSliceSize)
      }

      if trainable {
        calculateFilterGradientsInto(input,
                                      deltaSlicePtr: deltaSlicePtr,
                                      deltaSize: (rows: deltaRows, columns: deltaCols),
                                      filterIndex: i,
                                      resultPtr: wGradStorage.pointer + i * inputDepth * fSliceSize)
      }
    }

    // Bias gradients: sum each delta depth slice
    let biasStorage = TensorStorage.create(count: delta.size.depth)
    for d in 0..<delta.size.depth {
      let slicePtr = delta.storage.pointer + d * deltaSliceSize
      biasStorage[d] = NumSwiftFlat.sum(slicePtr, count: deltaSliceSize)
    }
    let biasesTensor = Tensor(storage: biasStorage, size: biases.size)
    biasesTensor.label = "transconv2d-bias"

    let inputTensor = Tensor(storage: inputGradStorage, size: inputSize)
    let wSize = TensorSize(rows: fRows, columns: fCols, depth: filterCount * inputDepth)
    let weightsTensor = Tensor(storage: wGradStorage, size: wSize)

    return (inputTensor, weightsTensor, biasesTensor)
  }

  internal override func calculateFilterGradientsInto(_ input: Tensor,
                                                       deltaSlicePtr: TensorStorage.Pointer,
                                                       deltaSize: (rows: Int, columns: Int),
                                                       filterIndex: Int,
                                                       resultPtr: TensorStorage.Pointer) {
    let fSliceSize = filterSize.rows * filterSize.columns
    let inputSliceSize = inputSize.rows * inputSize.columns

    // Determine stride-padded filter shape
    let baseFilterInputSize = (rows: inputSize.rows, columns: inputSize.columns)
    let filterInputSize: (rows: Int, columns: Int)
    let extraPadRight: Int
    let extraPadBottom: Int
    let spShape: (rows: Int, columns: Int)
    if strides.0 > 1 {
      spShape = NumSwiftFlat.stridePadShape(inputSize: baseFilterInputSize, strides: strides)
      extraPadRight  = strides.0 - 1
      extraPadBottom = strides.1 - 1
      filterInputSize = (rows: spShape.rows + extraPadBottom,
                         columns: spShape.columns + extraPadRight)
    } else {
      spShape = baseFilterInputSize
      filterInputSize = baseFilterInputSize
      extraPadRight  = 0
      extraPadBottom = 0
    }

    // Determine signal (delta) size, padding for .same
    let signalSize: (rows: Int, columns: Int)
    let deltaNeedsPad = (padding == .same)
    if deltaNeedsPad {
      let padCalc = NumSwiftFlat.paddingCalculation(strides: (1, 1), padding: .same,
                                                     filterSize: filterSize,
                                                     inputSize: (rows: outputSize.rows, columns: outputSize.columns))
      signalSize = (rows: outputSize.rows + padCalc.top + padCalc.bottom,
                   columns: outputSize.columns + padCalc.left + padCalc.right)
    } else {
      signalSize = deltaSize
    }

    // Buffer for stride-padded input must be sized to the stride-padded shape, not the original input shape
    let stridePaddedFilterSize = spShape.rows * spShape.columns
    let paddedFilterSize = filterInputSize.rows * filterInputSize.columns
    let paddedDeltaSize = signalSize.rows * signalSize.columns

    let stridePaddedFilterBuf = TensorStorage.create(count: stridePaddedFilterSize)
    let paddedFilterBuf = TensorStorage.create(count: paddedFilterSize)
    let paddedDeltaBuf = TensorStorage.create(count: paddedDeltaSize)

    // Pad signal (delta slice) once for all input depth iterations
    if deltaNeedsPad {
      let padCalc = NumSwiftFlat.paddingCalculation(strides: (1, 1), padding: .same,
                                                     filterSize: filterSize,
                                                     inputSize: (rows: outputSize.rows, columns: outputSize.columns))
      NumSwiftFlat.zeroPad1D(signal: deltaSlicePtr,
                              result: paddedDeltaBuf.pointer,
                              padding: NumSwiftPadding(top: padCalc.top, left: padCalc.left,
                                                       right: padCalc.right, bottom: padCalc.bottom),
                              inputSize: (rows: outputSize.rows, columns: outputSize.columns))
    } else {
      paddedDeltaBuf.pointer.update(from: deltaSlicePtr, count: paddedDeltaSize)
    }

    for i in 0..<inputSize.depth {
      let filterSrcPtr = input.storage.pointer + i * inputSliceSize

      if strides.0 > 1 {
        // Stride-pad the input depth slice
        NumSwiftFlat.stridePad1D(signal: filterSrcPtr,
                                  result: stridePaddedFilterBuf.pointer,
                                  strides: strides,
                                  signalSize: baseFilterInputSize)
        // Zero-pad right and bottom
        NumSwiftFlat.zeroPad1D(signal: stridePaddedFilterBuf.pointer,
                                result: paddedFilterBuf.pointer,
                                padding: NumSwiftPadding(top: 0, left: 0,
                                                         right: extraPadRight, bottom: extraPadBottom),
                                inputSize: spShape)
      } else {
        paddedFilterBuf.pointer.update(from: filterSrcPtr, count: inputSliceSize)
      }

      NumSwiftFlat.conv2d(signal: paddedDeltaBuf.pointer,
                           filter: paddedFilterBuf.pointer,
                           result: resultPtr + i * fSliceSize,
                           strides: (1, 1),
                           padding: .valid,
                           filterSize: filterInputSize,
                           inputSize: signalSize)
    }
  }
  
  internal override func conv(_ input: Tensor) -> TensorStorage {
    let outRows = outputSize.rows
    let outCols = outputSize.columns
    let outSliceSize = outRows * outCols
    let inRows = inputSize.rows
    let inCols = inputSize.columns
    
    var filterOutputs = [Tensor.Value?](repeating: nil, count: filterCount)
    
    for i in 0..<input.size.depth {
      let localInput = input.depthSlice(i)
      var workingInput = NumSwiftFlat.stridePad(signal: localInput, strides: strides,
                                                 inputSize: (rows: inRows, columns: inCols))
      
      let spShape = (strides.rows > 1 || strides.columns > 1)
        ? NumSwiftFlat.stridePadShape(inputSize: (rows: inRows,
                                                  columns: inCols),
                                      strides: strides)
        : (rows: inRows, columns: inCols)
      
      let inputRowsDiff = Double(outRows) - Double(spShape.rows)
      let inputColsDiff = Double(outCols) - Double(spShape.columns)
      
      let paddingTop = Int(ceil(inputRowsDiff / Double(2)))
      let paddingBottom = Int(floor(inputRowsDiff / Double(2)))
      let paddingLeft = Int(ceil(inputColsDiff / Double(2)))
      let paddingRight = Int(floor(inputColsDiff / Double(2)))
      
      let numPadding = NumSwiftPadding(top: paddingTop, left: paddingLeft,
                                       right: paddingRight, bottom: paddingBottom)
      
      workingInput = NumSwiftFlat.zeroPad(signal: workingInput,
                                          padding: numPadding,
                                          inputSize: spShape)
      
      let newRows = spShape.rows + paddingTop + paddingBottom
      let newColumns = spShape.columns + paddingLeft + paddingRight
      
      for f in 0..<filterCount {
        let kernel = filters[f].depthSlice(i)
        
        var grad = self.device.conv2d(signal: workingInput,
                                      filter: kernel,
                                      strides: (1,1),
                                      padding: .same,
                                      filterSize: filterSize,
                                      inputSize: (rows: newRows, columns: newColumns),
                                      outputSize: nil)
        
        if let existing = filterOutputs[f] {
          grad = grad + existing
        }
        
        filterOutputs[f] = grad
      }
    }
    
    // Assemble flat result
    let resultStorage = TensorStorage.create(count: outSliceSize * filterCount)
    for f in 0..<filterCount {
      if let output = filterOutputs[f] {
        var finalOutput = output
        if biasEnabled {
          finalOutput = finalOutput + biases.storage[f]
        }
        let start = f * outSliceSize
        for j in 0..<min(finalOutput.count, outSliceSize) {
          resultStorage[start + j] = finalOutput[j]
        }
      }
    }
    
    return resultStorage
  }
  
}
