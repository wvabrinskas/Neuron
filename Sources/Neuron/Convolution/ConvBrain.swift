//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/19/22.
//

import Foundation
import NumSwift
import Logger

public class ConvBrain: Logger {
  public var logLevel: LogLevel = .low
  public var loss: [Float] = []
  
  private var bias: Float
  private var inputSize: TensorSize
  private lazy var fullyConnected: Brain = {
    let b = Brain(learningRate: learningRate,
                  lossFunction: .crossEntropy,
                  initializer: initializer.type)

    b.addInputs(0) //can be some arbitrary number will update later
    b.replaceOptimizer(optimizer)
    b.logLevel = logLevel
    return b
  }()
  
  private var learningRate: Float
  private let flatten: Flatten = .init()
  private var lobes: [ConvolutionalSupportedLobe] = []
  private let epochs: Int
  private let batchSize: Int
  private let optimizer: OptimizerFunction?
  private var compiled: Bool = false
  private var previousFlattenedCount: Int = 0
  private var initializer: Initializer
  
  public init(epochs: Int,
              learningRate: Float,
              bias: Float = 1.0,
              inputSize: TensorSize,
              batchSize: Int,
              optimizer: Optimizer? = nil,
              initializer: InitializerType = .heNormal) {
    self.epochs = epochs
    self.learningRate = learningRate
    self.inputSize = inputSize
    self.batchSize = batchSize
    self.optimizer = optimizer?.get(learningRate: learningRate)
    self.initializer = initializer.build()
    self.bias = bias
  }
  
  public func addConvolution(filterSize: TensorSize = (3,3,3),
                             filterCount: Int) {
    
    //if we have a previous layer calculuate the new depth else use the input size depth
    let incomingSize = lobes.last?.outputSize ?? inputSize
    
    let filterDepth = incomingSize.depth
    let filter = (filterSize.rows, filterSize.columns, filterDepth)
    
    let model = ConvolutionalLobeModel(inputSize: incomingSize,
                                       activation: .reLu,
                                       bias: bias,
                                       filterSize: filter,
                                       filterCount: filterCount)
    
    let lobe = ConvolutionalLobe(model: model,
                                 learningRate: learningRate,
                                 optimizer: optimizer,
                                 initializer: initializer)
    lobes.append(lobe)
  }
  
  public func addMaxPool() {
    let inputSize = lobes.last?.outputSize ?? inputSize

    let model = PoolingLobeModel(inputSize: inputSize)
    let lobe = PoolingLobe(model: model)
    lobes.append(lobe)
  }
  
  public func addDenseNormal(_ count: Int,
                             rate: Float = 0.1,
                             momentum: Float = 0.99) {
    let bnModel = NormalizedLobeModel(nodes: count,
                                      activation: .reLu,
                                      bias: 0,
                                      momentum: momentum,
                                      normalizerLearningRate: rate)
    fullyConnected.add(bnModel)
  }

  public func feed(data: ConvTrainingData) -> [Float] {
    return feedInternal(input: data, training: false)
  }
  
  public func train(data: DatasetData,
                    epochCompleted: ((_ epoch: Int) -> ())? = nil,
                    completed: ((_ loss: [Float]) -> ())? = nil) {
    
    guard compiled else {
      self.log(type: .error, priority: .alwaysShow, message: "Please call compile() before training")
      return
    }
    
    self.log(type: .success, priority: .alwaysShow, message: "Training started.....")
    
    let training = Array(data.training[0...2000])
    let trainingData = training.batched(into: batchSize)
    
    let val = Array(data.val[0...2000])
    let validationData = val.batched(into: batchSize)

    for e in 0..<epochs {
      
      var b = 0
      for batch in trainingData {
        let batchLoss = trainOn(batch: batch)
        loss.append(batchLoss)
        b += 1
        
        if b % 5 == 0 {
          if let val = validationData.randomElement() {
            let valLoss = self.validateOn(batch: val)
            self.log(type: .message, priority: .low, message: "validation loss: \(valLoss)")
          }
        }
        
        self.log(type: .message, priority: .low, message: "training loss: \(batchLoss)")
        self.log(type: .message, priority: .low, message: "accuracy: \(fullyConnected.accuracy)")

      }
      
      
      self.log(type: .message, priority: .alwaysShow, message: "epoch: \(e)")
      epochCompleted?(e)
    }
    
    completed?(loss)
  }
  
  public func addDense(_ count: Int, activation: Activation = .reLu) {
    fullyConnected.add(LobeModel(nodes: count, activation: activation))
  }
  
  public func compile() {
    fullyConnected.compile()
    self.compiled = true && fullyConnected.compiled
  }
  
  private func validateOn(batch: [ConvTrainingData]) -> Float {
    var lossOnBatch: Float = 0

    for b in 0..<batch.count {
      let trainable = batch[b]
      let out = self.feedInternal(input: trainable, training: false)
      let loss = self.fullyConnected.loss(out, correct: trainable.label)
      
      lossOnBatch += loss / Float(batch.count)
    }
    
    return lossOnBatch
  }
  
  private func trainOn(batch: [ConvTrainingData]) -> Float {
    //zero gradients at the start of training on a batch
    zeroGradients()
    
    var lossOnBatch: Float = 0
    
    //TODO: figure out a way to perform this concurrently
    // maybe a state that holds all the variables?
    for b in 0..<batch.count {
      let trainable = batch[b]
      
      let out = self.feedInternal(input: trainable, training: true)
      
      fullyConnected.calculateAccuracy(out, label: trainable.label)
      
      let loss = self.fullyConnected.loss(out, correct: trainable.label)
      
      let outputDeltas = self.fullyConnected.getOutputDeltas(outputs: out,
                                                             correctValues: trainable.label)
      
      backpropagate(deltas: outputDeltas)

      lossOnBatch += loss / Float(batch.count)
    }
        
    adjustWeights(batchSize: batch.count)
    
    optimizer?.step()
    
    return lossOnBatch
  }
  
  internal func adjustWeights(batchSize: Int) {
    fullyConnected.adjustWeights(batchSize: batchSize)
    
    lobes.concurrentForEach { element, index in
      element.adjustWeights(batchSize: batchSize)
    }
    
   // lobes.forEach { $0.adjustWeights(batchSize: batchSize) }
  }
  
  internal func backpropagate(deltas: [Float]) {
    let backpropBrain = fullyConnected.backpropagate(with: deltas)
    let firstLayerDeltas = backpropBrain.firstLayerDeltas
    
    //backprop conv
    let reversedLobes = lobes.reversed()
    
    var newDeltas = flatten.backpropagate(deltas: firstLayerDeltas)
    
    reversedLobes.forEach { lobe in
      newDeltas = lobe.calculateGradients(with: newDeltas)
    }
  }
  
  internal func feedInternal(input: ConvTrainingData, training: Bool) -> [Float] {
    fullyConnected.trainable = training

    var out = input.data
    
    //feed all the standard lobes
    lobes.forEach { lobe in
      let newOut = lobe.feed(inputs: out, training: training)
      out = newOut
    }
    
    //flatten outputs
    let flat = flatten.feed(inputs: out, training: training)
    //feed to fully connected
    
    if flat.count != previousFlattenedCount {
      fullyConnected.replaceInputs(flat.count)
      previousFlattenedCount = flat.count
    }
    
    let result = fullyConnected.feed(input: flat)
    return result
  }
  
  public func zeroGradients() {
    lobes.forEach { $0.zeroGradients() }
    fullyConnected.zeroGradients()
  }
  
  public func clear() {
    loss.removeAll()
    lobes.forEach { $0.clear() }
    fullyConnected.clear()
  }
}
