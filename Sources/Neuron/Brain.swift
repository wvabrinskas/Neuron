//
//  Brain.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/6/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation
import Logger
import GameplayKit
import NumSwift
import Combine

public class Brain: Logger, Trainable, MetricCalculator {
  public typealias TrainableDatasetType = TrainingData
  
  public private(set) var visualizer: NetworkVisualizer?

  public var metricsToGather: Set<Metric> = []

  /// The verbosity of the printed logs
  public var logLevel: LogLevel = .none
  
  public var epochs: Int
  
  public var learningRate: Float
  
  public var loss: [Float] = []
  
  public var lobes: [Lobe] = []
  
  public var trainable: Bool = true
  
  public var metrics: [Metric: Float] = [:]
  
  /// If the brain object has been compiled and linked properly
  public private(set) var compiled: Bool = false
  
  /// The function to use to calculate the loss of the network
  private var lossFunction: LossFunction
  
  /// The threshold to compare the validation error with to determine whether or not to stop training.
  /// Default: 0.001. A number between 0 and 0.1 is usually accepted
  private var lossThreshold: Float
  
  /// The initializer to generate the layer weights
  private var initializer: InitializerType
  
  internal var model: ExportModel?
  
  /// Initialized layer weights for unit test purposes only
  internal var layerWeights: [[[Float]]] = []
  
  private var descent: GradientDescent
  
  private var optimizer: OptimizerFunction?
  private var optimizerType: Optimizer?
  
  internal var outputLayer: [Neuron] { self.lobes.last?.neurons ?? [] }
  
  internal var totalCorrectGuesses: Int = 0
  internal var totalGuesses: Int = 0

  internal var weightConstraints: ClosedRange<Float>? = nil
  
  /// Creates a Brain object that manages a network of Neuron objects
  /// - Parameters:
  ///   - model: Optional model to build the network
  ///   - learningRate: The rate at which the network will adjust its weights
  ///   - epochs: the number of times to train
  ///   - lossFunction: The function used to calculate loss at an epoch
  ///   - lossThreshold: The threshold to stop training to prevent overfitting 0 - 1
  ///   - initializer: The weight initializer algoriUthm
  ///   - descent: The gradient descent type
  public init(model: ExportModel? = nil,
              learningRate: Float = 0.001,
              epochs: Int = 10,
              lossFunction: LossFunction = .meanSquareError,
              lossThreshold: Float = 0.001,
              initializer: InitializerType = .xavierNormal,
              descent: GradientDescent = .sgd,
              weightConstraints: ClosedRange<Float>? = nil,
              metrics: Set<Metric> = []) {
    
    self.learningRate = learningRate
    self.lossFunction = lossFunction
    self.lossThreshold = lossThreshold
    self.epochs = epochs
    self.initializer = initializer
    self.descent = descent
    self.model = model
    self.weightConstraints = weightConstraints
    self.metricsToGather = metrics
  }
  
  public init?(model: PretrainedModel,
               epochs: Int,
               lossFunction: LossFunction = .crossEntropy,
               lossThreshold: Float = 0.001,
               initializer: InitializerType = .xavierNormal,
               descent: GradientDescent = .sgd) {
    
    do {
      guard let model = try model.getModel().get() else {
        Self.log(type: .error, priority: .alwaysShow, message: "Could not build model")
        return nil
      }
      
      self.learningRate = model.learningRate
      self.epochs = epochs
      self.lossFunction = lossFunction
      self.lossThreshold = lossThreshold
      self.initializer = initializer
      self.model = model
      self.descent = descent
      
    } catch {
      Self.log(type: .error, priority: .alwaysShow, message: error.localizedDescription)
      return nil
    }
    
  }
  
  /// Replaces the weights in the network
  /// - Parameter weights: the weights to replace the existing weights with
  public func replaceWeights(weights: [[[Float]]]) {
    for l in 0..<self.lobes.count {
      let lobe = self.lobes[l]
      
      let weightsForLayer = weights[l]
      
      for n in 0..<lobe.neurons.count {
        let neuron = lobe.neurons[n]
        let weightsForNeuron = weightsForLayer[n]
        neuron.replaceWeights(weights: weightsForNeuron)
      }
    }

    self.layerWeights = lobes.map { $0.weights() }
  }
  
  /// Adds an optimizer to the gradient descent algorithm.
  /// Be sure to call this before calling `compile()`
  /// - Parameter optimizer: The optimizer type to add
  public func add(optimizer: Optimizer) {
    self.optimizer = optimizer.get(learningRate: self.learningRate)
    self.optimizerType = optimizer
  }
  
  public func visualize(_ vis: NetworkVisualizer) {
    visualizer = vis
  }
  
  /// Returns a model that can be imported later
  /// - Returns: The ExportModel that describes the network
  public func exportModel() -> ExportModel {
    
    let learningRate = self.learningRate
    var layers: [Layer] = []
    //link all the layers generating the matrix
    for i in 0..<self.lobes.count {
      let lobe = self.lobes[i]
      
      var weights: [[Float]] = []
      var biases: [Float] = []
      var biasWeights: [Float] = []
      
      lobe.neurons.forEach { (neuron) in
        weights.append(neuron.weights)
        biases.append(neuron.bias)
        biasWeights.append(neuron.biasWeight)
      }
      //set to 0
      if lobe.layer == .input {
        biasWeights = [Float](repeating: 0, count: lobe.neurons.count)
        biases = [Float](repeating: 0, count: lobe.neurons.count)
        weights = [[Float]](repeating: [0], count: lobe.neurons.count)
      }
      
      var layer = Layer(activation: lobe.activation,
                        nodes: lobe.neurons.count,
                        weights: weights,
                        type: lobe.layer,
                        bias: biases,
                        biasWeights: biasWeights,
                        normalize: lobe.isNormalized)
      
      if let normalLobe = lobe as? NormalizedLobe {
        layer = Layer(activation: lobe.activation,
                      nodes: lobe.neurons.count,
                      weights: weights,
                      type: lobe.layer,
                      bias: biases,
                      biasWeights: biasWeights,
                      normalize: lobe.isNormalized,
                      batchNormalizerParams: normalLobe.normalizerLearningParams)
      }
      
      layers.append(layer)
    }
    
    let model = ExportModel(layers: layers,
                            learningRate: learningRate,
                            optimizer: optimizerType)
    return model
  }
  
  /// Returns a model that can be imported later
  /// - Returns: The url that can be downloaded to get the Smodel file
  public func exportModelURL() -> URL? {
    return ExportManager.getModel(filename: "model", model: self.exportModel())
  }
  
  //TODO: Refactor to support easy layer adding like the ConvBrain
  /// Adds a layer to the neural network
  /// - Parameter model: The lobe model describing the layer to be added
  public func add(_ model: LobeDefinition) {
    guard lobes.count >= 1 else {
      fatalError("ERROR: Please call addInputs(_ count: Int) before adding a hidden layer")
    }
    
    var lobe = Lobe(model: model,
                    learningRate: learningRate)
    
    if let bnModel = model as? NormalizedLobeModel  {
      lobe = NormalizedLobe(model: bnModel,
                            learningRate: learningRate)
    }
    
    self.lobes.append(lobe)
  }
  
  public func replaceOptimizer(_ optimizer: OptimizerFunction?) {
    self.optimizer = optimizer
  }
  
  private func compileWithModel() {
    guard let model = self.model else {
      return
    }
    
    self.optimizer = model.optimizer?.get(learningRate: model.learningRate)
    //go through each layer
    model.layers.forEach { (layer) in
      
      var neurons: [Neuron] = []
      
      //go through each node in layer
      for i in 0..<layer.nodes {
        precondition(i < layer.weights.count && i < layer.bias.count && i < layer.biasWeights.count)
        
        let weights = layer.weights[i]
        let bias = layer.bias[i]
        let biasWeight = layer.biasWeights[i]
        
        //map each weight
        let dendrites = weights.map({ NeuroTransmitter(weight: $0) })
        
        let nucleus = Nucleus(learningRate: self.learningRate, bias: bias)
        
        let neuron = Neuron(inputs: dendrites,
                            nucleus: nucleus,
                            activation: layer.activation,
                            optimizer: optimizer)
        
        neuron.layer = layer.type
        neuron.biasWeight = biasWeight
        
        neurons.append(neuron)
      }
      
      var lobe = Lobe(neurons: neurons,
                      activation: layer.activation)
      
      if layer.normalize, let bn = layer.batchNormalizerParams {
        lobe = NormalizedLobe(neurons: neurons,
                              activation: layer.activation,
                              beta: bn.beta,
                              gamma: bn.gamma,
                              momentum: bn.momentum,
                              learningRate: model.learningRate,
                              batchNormLearningRate: bn.learningRate,
                              movingMean: bn.movingMean,
                              movingVariance: bn.movingVariance)
      }
      
      lobe.layer = layer.type
      
      self.lobes.append(lobe)
    }
    
    self.layerWeights = self.lobes.map { $0.weights() }
    
    self.compiled = true
    self.model = nil
  }
  
  /// Connects all the lobes together in the network builing the complete network 
  public func compile() {
    guard self.compiled == false else {
      return
    }
    
    guard self.model == nil else {
      self.compileWithModel()
      return
    }
    
    guard lobes.count > 0 else {
      fatalError("no lobes to connect bailing out.")
    }
    
    //keep the first layer since there's nothing we need to do with it
    self.layerWeights = [self.layerWeights[safe: 0]  ?? []]
    
    //link all the layers generating the matrix
    for i in 1..<lobes.count {
      let lobe = self.lobes[i]
      let layerType: LayerType = i + 1 == lobes.count ? .output : .hidden
      
      let inputNeuronGroup = self.lobes[i-1].outputCount
      
      let compileModel = LobeCompileModel.init(inputNeuronCount: inputNeuronGroup,
                                               layerType: layerType,
                                               fullyConnected: true,
                                               weightConstraint: self.weightConstraints,
                                               initializer: self.initializer,
                                               optimizer: self.optimizer)
      
      let weights = lobe.compile(model: compileModel)
      self.layerWeights.append(weights)
    }
    
    compiled = true
  }
  
  public func addInputs(_ count: Int) {
    let compileModel = LobeCompileModel.init(inputNeuronCount: 0,
                                             layerType: .input,
                                             fullyConnected: false,
                                             weightConstraint: self.weightConstraints,
                                             initializer: self.initializer,
                                             optimizer: self.optimizer)
    
    //first layer weight initialization with 0 since it's just the input layer
    let inputLayer = Lobe(model: LobeModel(nodes: count),
                          learningRate: self.learningRate)
    
    let weights = inputLayer.compile(model: compileModel)
    self.layerWeights.append(weights)
    
    if self.lobes.first != nil {
      self.lobes[0] = inputLayer
    } else {
      self.lobes.append(inputLayer)
    }
  }
  
  public func replaceInputs(_ count: Int) {
    addInputs(count)
    recompile()
  }
  
  private func recompile() {
    compiled = false
    compile()
  }
  
  /// Supervised training function
  /// - Parameters:
  ///   - data: The training data object containing the expected values and the training data
  ///   - validation: The validation data object containing the expected values and the validation data
  ///   - epochCompleted: Called when an epoch is completed with the current epoch
  ///   - complete: Called when training is completed
  public func train(dataset: InputData,
                    epochCompleted: ((Int, [Metric : Float]) -> ())? = nil,
                    complete: (([Metric : Float]) -> ())? = nil) {
    
    let data = dataset.training
    let validation = dataset.validation
        
    let trainingStartDate = Date()
    
    guard data.count > 0 else {
      print("data must contain some data")
      complete?(metrics)
      return
    }
    
    guard compiled == true else {
      complete?(metrics)
      print("please run compile() on the Brain object before training")
      return
    }
    
    let mixedData = data
    var setBatches: Bool = false
    var batches: [[TrainingData]] = []
    
    let valBatches = validation.batched(into: 10)

    for i in 0..<epochs {
      self.trainable = true

      let epochStartDate = Date()
      
      self.log(type: .message, priority: .medium, message: "epoch: \(i)")
      
      var epochLoss: Float = 0
      
      switch self.descent {
      case .bgd:
        let loss = self.trainOn(data)
        epochLoss = loss

      case .sgd:
        if setBatches == false {
          setBatches = true
          batches = mixedData.batched(into: 1)
        }
        
        var lossOnBatches: Float = 0
        
        batches.forEach { (batch) in
          lossOnBatches += self.trainOn(batch) / Float(batches.count)
        }
        
        epochLoss = lossOnBatches
        
      case .mbgd(let size):
        if setBatches == false {
          setBatches = true
          batches = mixedData.batched(into: size)
        }
        
        var lossOnBatches: Float = 0
        
        batches.forEach { (batch) in
          lossOnBatches += self.trainOn(batch) / Float(batches.count)
        }
        
        epochLoss = lossOnBatches
      }
      
      loss.append(epochLoss)
      addMetric(value: epochLoss, key: .loss)
      
      self.log(type: .message,
               priority: .low,
               message: "    loss: \(metrics[.loss] ?? 0)")
      self.log(type: .message,
               priority: .low,
               message: "    accuracy: \(metrics[.accuracy] ?? 0)")
    

      self.log(type: .message,
               priority: .high,
               message: "epoch completed time: \(Date().timeIntervalSince(epochStartDate))")
      
      //feed validation data through the network
      if let validationData = valBatches.randomElement(), validation.count > 0 {
        self.trainable = false
        
        let errorForValidation = self.validateOn(validationData)
        
        addMetric(value: errorForValidation, key: .valLoss)
        
        self.log(type: .message,
                 priority: .low,
                 message: "val loss: \(errorForValidation)")
        
        //if validation error is greater than previous then we are complete with training
        //bail out to prevent overfitting
        let threshold: Float = self.lossThreshold
        if errorForValidation <= threshold {
          
          self.log(type: .success, priority: .alwaysShow, message: "SUCCESS: training is complete...")
          self.log(type: .message, priority: .alwaysShow, message: "Metrics: \(self.metrics)")
          
          self.log(type: .success,
                   priority: .high,
                   message: "training completed time: \(Date().timeIntervalSince(trainingStartDate))")
          
          
          complete?(metrics)
          return
        }
      }
      
      epochCompleted?(i, metrics)
    }
    
    self.log(type: .success,
             priority: .low,
             message: "training completed time: \(Date().timeIntervalSince(trainingStartDate))")
    
    self.log(type: .message, priority: .low, message: "Loss: \(self.metrics[.loss] ?? 0)")
    
    //false because the training wasnt completed with validation
    complete?(metrics)
  }
  
  public func validateOn(_ batch: [TrainingData]) -> Float {
    self.trainable = false
    
    var batchLoss: Float = 0

    batch.forEach { vData in
      //feed the data through the network
      let output = self.feed(input: vData.data)
      
      calculateAccuracy(output, label: vData.correct, binary: outputLayer.count == 1)
      
      batchLoss += self.loss(output, correct: vData.correct) / Float(batch.count)
    }
    
    return batchLoss
  }
  
  public func trainOn(_ batch: [TrainingData]) -> Float {
    self.trainable = true
    self.zeroGradients()
    
    var batchLoss: Float = 0

    batch.forEach { tData in
      //feed the data through the network
      let output = self.feed(input: tData.data)
      
      calculateAccuracy(output, label: tData.correct, binary: outputLayer.count == 1)
          
      batchLoss += self.loss(output, correct: tData.correct) / Float(batch.count)
      
      //set the output errors for set
      let deltas = self.getOutputDeltas(outputs: output,
                                        correctValues: tData.correct)
  
      self.backpropagate(with: deltas)
    }
         
    self.adjustWeights(batchSize: batch.count)
    
    self.optimizer?.step()

    return batchLoss
  }

  /// Clears the whole network and resets all the weights to a random value
  public func clear() {
    metrics.removeAll()
    //clears the whole matrix
    self.lobes.forEach { (lobe) in
      lobe.clear()
    }
  }
  
  public func weights() -> [[[Float]]] {
    lobes.map { $0.weights() }
  }
  
  /// Feed-forward through the network to get the result
  /// - Parameters:
  ///   - input: the input array of floats
  /// - Returns: The result of the feed forward through the network as an array of Floats
  public func feed(input: [Float]) -> [Float] {
    let output = self.feedInternal(input: input)
    return output
  }
  
  /// Update the settings for each lobe and each neuron
  /// - Parameter nucleus: Object that describes the network behavior
  public func updateNucleus(_ nucleus: Nucleus) {
    self.lobes.forEach { (lobe) in
      lobe.updateNucleus(nucleus)
    }
  }
  
  /// Exports the loss data as a CSV
  /// - Parameter filename: Name of the file to save and export. defaults to `loss-{timeIntervalSince1970}`
  /// - Returns: The url of the exported file if successful.
  public func exportLoss(_ filename: String? = nil) -> URL? {
    let name = filename ?? "loss-\(Int(Date().timeIntervalSince1970))"
    return ExportManager.getCSV(filename: name, loss)
  }
  
  /// Feeds the network internally preparing for output or training
  /// - Parameter input: the input data to feed the network
  internal func feedInternal(input: [Float]) -> [Float] {
    var x = input
    
    for i in 0..<self.lobes.count {
      let currentLayer = self.lobes[i]
      let newInputs: [Float] = currentLayer.feed(inputs: x, training: self.trainable)

      x = newInputs
    }
    
    DispatchQueue.main.async {
      self.visualizer?.visualize(brain: self)
    }
    
    return x
  }

  internal func loss(_ predicted: [Float], correct: [Float]) -> Float {
    self.lossFunction.calculate(predicted, correct: correct)
  }
  
  //calculates the error at the output layer w.r.t to the weights.
  internal func getOutputDeltas(outputs: [Float], correctValues: [Float]) -> [Float] {
    guard correctValues.count == self.outputLayer.count,
          outputs.count == self.outputLayer.count else {
      
      self.log(type: .error,
               priority: .alwaysShow,
               message: "Error: correct data count does not match ouput node count, bailing out")
      return []
    }
    
    var outputErrors: [Float] = []
    
    for i in 0..<self.outputLayer.count {
      let target = correctValues[i]
      let predicted = outputs[i]
      
      let delta = self.lossFunction.derivative(predicted, correct: target)
      
      outputErrors.append(delta)
    }
    
    return outputErrors
  }
  
  internal func gradients() -> [[[Float]]] {
    return self.lobes.map { $0.gradients() }
  }

  internal func adjustWeights(batchSize: Int) {
    self.lobes.forEach { $0.adjustWeights(batchSize: batchSize) }
  }
  
  @discardableResult
  internal func backpropagate(with deltas: [Float]) -> (firstLayerDeltas: [Float], gradients: [[[Float]]]) {
    //reverse so we can loop through from the beggining of the array starting at the output node
    let reverse: [Lobe] = self.lobes.reversed()

    guard reverse.count > 0 else {
      return ([], [])
    }
        
    var updatingDeltas = deltas
    var gradients: [[[Float]]] = []
    
    //subtracting 1 because we dont need to propagate through to the weights in the input layer
    //those will always be 0 since no computation happens at the input layer
    for i in 0..<reverse.count - 1 {
      let currentLobe = reverse[i]
      let previousLobe = reverse[i + 1]
      
      //calculate gradients for current layer with previous layer errors aka deltas
      let newGradients = currentLobe.calculateGradients(with: updatingDeltas)

      //current lobe is calculating the deltas for the previous lobe
      updatingDeltas = currentLobe.calculateDeltasForPreviousLayer(incomingDeltas: updatingDeltas,
                                                                   previousLayerCount: previousLobe.outputCount)
      
      gradients.append(newGradients)
    }
    
    return (updatingDeltas, gradients)
  }
  
  internal func zeroGradients() {
    self.lobes.forEach { lobe in
      lobe.zeroGradients()
    }
  }
}
