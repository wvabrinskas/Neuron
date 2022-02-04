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

public class Brain: Logger {
  /// The verbosity of the printed logs
  public var logLevel: LogLevel = .none
  
  /// Number of times to run the training data set
  public var epochs: Int
  
  public var learningRate: Float
  
  /// The loss function data from training to be exported
  public var loss: [Float] = []
  
  /// Neuron matrix in the existing brain object
  public var lobes: [Lobe] = []
  
  public var trainable: Bool = true
  
  /// If the brain object has been compiled and linked properly
  private(set) var compiled: Bool = false
  
  /// The function to use to calculate the loss of the network
  private var lossFunction: LossFunction
  
  /// The threshold to compare the validation error with to determine whether or not to stop training.
  /// Default: 0.001. A number between 0 and 0.1 is usually accepted
  private var lossThreshold: Float
  
  /// Output modifier for the output layer. ex. softmax
  private var outputModifier: OutputModifier? = nil
  
  /// The previous set of validation errors
  private var previousValidationErrors: [Float] = []
  
  /// The initializer to generate the layer weights
  private var initializer: Initializers
  
  internal var model: ExportModel?
  
  /// Initialized layer weights for unit test purposes only
  internal var layerWeights: [[Float]] = []
  
  /// Distribution for initialization
  internal static let dist = NormalDistribution()
  
  private var descent: GradientDescent
  
  private var optimizer: OptimizerFunction?
  private var optimizerType: Optimizer?
  
  private var outputLayer: [Neuron] { self.lobes.last?.neurons ?? [] }
        
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
              learningRate: Float,
              epochs: Int,
              lossFunction: LossFunction = .meanSquareError,
              lossThreshold: Float = 0.001,
              initializer: Initializers = .xavierNormal,
              descent: GradientDescent = .sgd,
              weightConstraints: ClosedRange<Float>? = nil) {
    
    self.learningRate = learningRate
    self.lossFunction = lossFunction
    self.lossThreshold = lossThreshold
    self.epochs = epochs
    self.initializer = initializer
    self.descent = descent
    self.model = model
    self.weightConstraints = weightConstraints
  }
  
  public init?(model: PretrainedModel,
               epochs: Int,
               lossFunction: LossFunction = .crossEntropy,
               lossThreshold: Float = 0.001,
               initializer: Initializers = .xavierNormal,
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
  public func replaceWeights(weights: [[Float]]) {
    var i = 0
    var newWeights: [[Float]] = []
    self.lobes.forEach { (lobe) in
      
      lobe.neurons.forEach { (n) in
        var j = 0
        var newNodeWeights: [Float] = []
        n.replaceWeights(weights: weights[i])
        
        n.weights.forEach { weight in
          newNodeWeights.append(weight)
          j += 1
        }
        
        newWeights.append(newNodeWeights)
        i += 1
      }
    }
    self.layerWeights = newWeights
  }
  
  /// Adds an optimizer to the gradient descent algorithm.
  /// Be sure to call this before calling `compile()`
  /// - Parameter optimizer: The optimizer type to add
  public func add(optimizer: Optimizer) {
    self.optimizer = optimizer.get()
    self.optimizerType = optimizer
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
  /// Adds a layer to the neural network
  /// - Parameter model: The lobe model describing the layer to be added
  public func add(_ model: LobeModel) {
    var lobe = Lobe(model: model,
                    learningRate: self.learningRate)
    
    if model.normalize {
      guard let momentum = model.bnMomentum, let bnLearningRate = model.bnLearningRate else {
        fatalError("please provide a momentum and learning rate for the Batch Normalizer")
      }
      
      lobe = NormalizedLobe(model: model,
                            learningRate: self.learningRate,
                            momentum: momentum,
                            batchNormLearningRate: bnLearningRate)
    }
    
    self.lobes.append(lobe)
  }
  
  /// Adds an output modifier to the output layer
  /// - Parameter mod: The modifier to apply to the outputs
  public func add(modifier mod: OutputModifier) {
    self.outputModifier = mod
  }
  
  private func compileWithModel() {
    guard let model = self.model else {
      return
    }
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
                            optimizer: model.optimizer)
        
        neuron.layer = layer.type
        neuron.biasWeight = biasWeight
        
        neurons.append(neuron)
        self.layerWeights.append(weights)
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
    
    self.compiled = true
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
    self.layerWeights.removeAll()
    
    //link all the layers generating the matrix
    for i in 0..<lobes.count {
      if i > 0 {
        let lobe = self.lobes[i]
        let layerType: LobeModel.LayerType = i + 1 == lobes.count ? .output : .hidden
        
        let inputNeuronGroup = self.lobes[i-1].neurons
        
        let compileModel = LobeCompileModel.init(inputNeuronCount: inputNeuronGroup.count,
                                                 layerType: layerType,
                                                 fullyConnected: true,
                                                 weightConstraint: self.weightConstraints,
                                                 initializer: self.initializer,
                                                 optimizer: self.optimizer)
        
        let weights = lobe.compile(model: compileModel)
        self.layerWeights.append(contentsOf: weights)

      } else {
        
        let compileModel = LobeCompileModel.init(inputNeuronCount: 0,
                                                 layerType: .input,
                                                 fullyConnected: false,
                                                 weightConstraint: self.weightConstraints,
                                                 initializer: self.initializer,
                                                 optimizer: self.optimizer)
        
        //first layer weight initialization with 0 since it's just the input layer
        let inputLayer = self.lobes[i]
        let weights = inputLayer.compile(model: compileModel)
        self.layerWeights.append(contentsOf: weights)
      }
    }
    
    compiled = true
  }
  
  /// Supervised training function
  /// - Parameters:
  ///   - data: The training data object containing the expected values and the training data
  ///   - validation: The validation data object containing the expected values and the validation data
  ///   - epochCompleted: Called when an epoch is completed with the current epoch
  ///   - complete: Called when training is completed
  public func train(data: [TrainingData],
                    validation: [TrainingData] = [],
                    epochCompleted: ((_ epoch: Int) -> ())? = nil,
                    complete: ((_ passedValidation: Bool) -> ())? = nil) {
    
    previousValidationErrors.removeAll()
    
    let trainingStartDate = Date()
    
    guard data.count > 0 else {
      print("data must contain some data")
      complete?(false)
      return
    }
    
    guard compiled == true else {
      complete?(false)
      print("please run compile() on the Brain object before training")
      return
    }
    
    let mixedData = data//.randomize()
    var setBatches: Bool = false
    var batches: [[TrainingData]] = []

    for i in 0..<epochs {
      self.trainable = true

      let epochStartDate = Date()
      
      self.log(type: .message, priority: .low, message: "epoch: \(i)")
            
      switch self.descent {
      case .bgd:
        self.trainOnBatch(batch: data)
      case .sgd:
        guard let obj = mixedData.randomElement() else {
          return
        }
        
        self.trainOnBatch(batch: [obj])
        
      case.mbgd(let size):
        if setBatches == false {
          setBatches = true
          batches = mixedData.batched(into: size)
        }
        
        batches.forEach { (batch) in
          self.trainOnBatch(batch: batch)
        }
      }
      
      self.log(type: .message,
               priority: .high,
               message: "epoch completed time: \(Date().timeIntervalSince(epochStartDate))")
      
      //feed validation data through the network
      if let validationData = validation.randomElement(), validation.count > 0 {
        self.trainable = false
        
        let output = self.feed(input: validationData.data)
        let errorForValidation = self.loss(output,
                                           correct: validationData.correct)
        
//        self.log(type: .message, priority: .low, message: "val error at epoch \(i): \(errorForValidation) \(output) \(validationData.correct)")

        //if validation error is greater than previous then we are complete with training
        //bail out to prevent overfitting
        let threshold: Float = self.lossThreshold
        //only append if % 10 != 0
        let checkBatchCount = 5
        
        if i % checkBatchCount == 0 {
          
          self.log(type: .message, priority: .medium, message: "validating....")
          
          if self.averageError() <= threshold {
            
            self.log(type: .success, priority: .alwaysShow, message: "SUCCESS: training is complete...")
            self.log(type: .message, priority: .alwaysShow, message: "Loss: \(self.loss.last ?? 0)")
            
            self.log(type: .success,
                     priority: .high,
                     message: "training completed time: \(Date().timeIntervalSince(trainingStartDate))")
            
            
            complete?(true)
            return
          }
        }
        
        if previousValidationErrors.count == checkBatchCount {
          previousValidationErrors.removeFirst()
        }
        
        previousValidationErrors.append(errorForValidation)
      }
      
      epochCompleted?(i)
    }
    
    self.log(type: .success,
             priority: .low,
             message: "training completed time: \(Date().timeIntervalSince(trainingStartDate))")
    
    self.log(type: .message, priority: .low, message: "Loss: \(self.loss.last ?? 0)")
    
    //false because the training wasnt completed with validation
    complete?(false)
  }
  
  private func trainOnBatch(batch: [TrainingData]) {
    self.trainable = true
    self.zeroGradients()
    
    var losses: [Float] = []
    batch.forEach { tData in
      //feed the data through the network
      let output = self.feed(input: tData.data)
      
      let newLoss = self.loss(output, correct: tData.correct)
      losses.append(newLoss)
      
      //set the output errors for set
      let deltas = self.getOutputDeltas(outputs: output,
                                        correctValues: tData.correct)
      
      self.backpropagate(with: deltas)
    }
    
    let lossAvg = losses.sum / Float(losses.count)

    self.loss.append(lossAvg)
    
    self.log(type: .message, priority: .low, message: "loss at batch: \(lossAvg)")
    
    self.adjustWeights(batchSize: batch.count)
    self.optimizer?.step()
  }

  /// Clears the whole network and resets all the weights to a random value
  public func clear() {
    self.previousValidationErrors = []
    self.loss.removeAll()
    //clears the whole matrix
    self.lobes.forEach { (lobe) in
      lobe.clear()
    }
  }
  
  /// Feed-forward through the network to get the result
  /// - Parameters:
  ///   - input: the input array of floats
  /// - Returns: The result of the feed forward through the network as an array of Floats
  public func feed(input: [Float]) -> [Float] {
    let output = self.feedInternal(input: input)
    let modified = self.applyModifier(outputs: output)
    return modified
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
    return ExportManager.getCSV(filename: name, self.loss)
  }
  
  
  internal func averageError() -> Float {
    previousValidationErrors.sum / Float(previousValidationErrors.count)
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
    
    return x
  }
  
  /// Applies the output modifier
  /// - Parameter outputs: outputs to apply modifier to
  /// - Returns: the modified output given by the set modifier
  private func applyModifier(outputs: [Float]) -> [Float] {
    
    guard let mod = self.outputModifier else {
      return outputs
    }
    
    var modOut: [Float] = []
    
    for i in 0..<outputs.count {
      let modVal = mod.calculate(index: i, outputs: outputs)
      modOut.append(modVal)
    }
    
    return modOut
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
    
    reverse.first?.calculateGradients(with: updatingDeltas)
    
    var gradients: [[[Float]]] = []

    //subtracting 1 because we dont need to propagate through to the weights in the input layer
    //those will always be 0 since no computation happens at the input layer
    for i in 0..<reverse.count - 1 {
      let currentLobe = reverse[i]
      let previousLobe = reverse[i + 1]
      
      //incoming inputs are the new deltas for the current layer
      updatingDeltas = currentLobe.calculateDeltas(inputs: updatingDeltas,
                                                   previousLayerCount: previousLobe.neurons.count)
      
      //calculatte gradients for next layer since we calculated the deltas for this current one. This is in reverse so technically it's in order..
      let newGradients = previousLobe.calculateGradients(with: updatingDeltas)
      gradients.append(newGradients)
    }
    
    return (updatingDeltas, gradients)
  }
  
  internal func firstNonEmptyLayerDeltas() -> [Float]? {
    return lobes.first(where: { $0.deltas().count > 0 })?.deltas()
  }
  
  internal func zeroGradients() {
    self.lobes.forEach { lobe in
      lobe.zeroGradients()
    }
  }
}
