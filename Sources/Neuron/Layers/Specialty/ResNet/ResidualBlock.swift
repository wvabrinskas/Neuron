//
//  ResNetBlock.swift
//

import Foundation
import NumSwift

public final class ResidualBlock {
  var allLayers: [Layer] = []
  var outputSize: TensorSize = .init(array: [])
  let inputSize: TensorSize
  private var block: Sequential = .init()
  var shortcut: Sequential = .init()
  private var reluOut: ReLu = .init()
  private var config: ResidualBlockConfiguration
  
  public init(config: ResidualBlockConfiguration,
              inputSize: TensorSize) {
    self.inputSize = inputSize
    self.config = config
    
    build()
  }
  
  public func build() {
    let conv1 = Conv2d(filterCount: config.filters,
                       inputSize: inputSize,
                       strides: config.strides,
                       padding: .same,
                       filterSize: config.filterSize,
                       initializer: .heNormal,
                       biasEnabled: true)
    
    let batchNorm1 = BatchNormalize()
    
    let relu1 = ReLu()
    
    let conv2 = Conv2d(filterCount: config.filters,
                       strides: config.strides,
                       padding: .same,
                       filterSize: config.filterSize,
                       initializer: .heNormal,
                       biasEnabled: true)
    
    let batchNorm2 = BatchNormalize()
    
    block = .init(conv1, batchNorm1, relu1, conv2, batchNorm2)
    block.compile()
    
    // build shortcut
    // should have the same input size as conv1
    let shortcutConv = Conv2d(filterCount: config.filters,
                              inputSize: inputSize,
                              strides: config.strides,
                              padding: .same,
                              filterSize: config.filterSize,
                              initializer: .heNormal,
                              biasEnabled: true)
    
    let shortcutBatchNorm = BatchNormalize()
    
    shortcut = .init(shortcutConv, shortcutBatchNorm)
    shortcut.compile()
    
    reluOut.inputSize = batchNorm2.outputSize // SHOULD be the same dimensions as shortcut as well
    
    outputSize = reluOut.outputSize
  
    // maybe we dont include short cut layers and handle weight / gradient manually?
    allLayers = block.layers + shortcut.layers + [reluOut]
  }
  
  public func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    let blockForward = block(tensor, context: context)
    let shortcutForward = shortcut(tensor, context: context)
    let output = blockForward + shortcutForward
    
    let reluOutForward = reluOut.forward(tensor: output, context: context)
    
    
    // TODO figure out weight and gradient calculation here.
    let tensorContext = TensorContext { inputs, gradient in
      // backpropogation calculation
      let errorRelu = reluOutForward.gradients(delta: gradient)
      
      // this sum isnt correct because this asssumes it's a scalar value.
//      let errorReluInput = errorRelu.input.first?.sum(axis: -1) ?? Tensor()
//      
//      let blockGradient = blockForward.gradients(delta: errorReluInput)
//      let shortcutGradiet = shortcutForward.gradients(delta: errorReluInput)
      
      return (Tensor(), Tensor(), Tensor())
    }
    
    // forward calculation
    return Tensor(reluOutForward.value, context: tensorContext)
  }
}

