//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/19/22.
//

import Foundation
import Logger

public class ConvBrain: Logger {
  public var logLevel: LogLevel = .low
  
  private var inputSize: TensorSize
  private var fullyConnected: Brain
  private var learningRate: Float
  private let flatten: Flatten = .init()
  private var lobes: [ConvolutionalSupportedLobe] = []
  private let epochs: Int
  
  public init(epochs: Int,
              learningRate: Float,
              inputSize: TensorSize,
              fullyConnected: Brain) {
    self.epochs = epochs
    self.learningRate = learningRate
    self.inputSize = inputSize
    
    self.fullyConnected = fullyConnected
    self.fullyConnected.learningRate = learningRate
  }
  
  public func addFullyConnected(_ brain: Brain) {
    self.fullyConnected = brain
    brain.learningRate = learningRate
    if brain.compiled == false {
      brain.compile()
    }
  }
  
  public func addConvolution(bias: Float = 1.0,
                             filterSize: TensorSize = (3,3,3),
                             filterCount: Int) {
    let inputSize = lobes.last?.outputSize ?? inputSize
    
    var filter = filterSize
    if let filterDepth = lobes.last?.outputSize.depth {
      filter = (filterSize.rows, filterSize.columns, filterDepth)
    }
  
    let model = ConvolutionalLobeModel(inputSize: inputSize,
                                       activation: .reLu,
                                       bias: bias,
                                       filterSize: filter,
                                       filterCount: filterCount)
    
    let lobe = ConvolutionalLobe(model: model, learningRate: learningRate)
    lobes.append(lobe)
  }
  
  public func addMaxPool() {
    let inputSize = lobes.last?.outputSize ?? inputSize

    let model = PoolingLobeModel(inputSize: inputSize)
    let lobe = PoolingLobe(model: model)
    lobes.append(lobe)
  }
  
  public func feed(data: ConvTrainingData) -> [Float] {
    var out: [[[Float]]] = data.data
    
    lobes.forEach { lobe in
      let newOut = lobe.feed(inputs: out, training: false)
      out = newOut
    }
    
    let flat = flatten.feed(inputs: out)
    let brainOut = fullyConnected.feed(input: flat)
    return brainOut
  }
  
  public func zeroGradients() {
    lobes.forEach { $0.zeroGradients() }
    fullyConnected.zeroGradients()
  }
  
  public func clear() {
    lobes.forEach { $0.clear() }
    fullyConnected.clear()
  }
}
