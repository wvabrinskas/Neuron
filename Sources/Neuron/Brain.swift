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
  
  public var batchNormalizerLearningRate: Float?
  
  /// The loss function data from training to be exported
  public var loss: [Float] = []
  
  /// Neuron matrix in the existing brain object
  public var lobes: [Lobe] = []
  
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
  
  private var optimizer: Optimizer?
  
  private var descents: [[Float]] = []
  
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
              batchNormalizerLearningRate: Float? = nil,
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
    self.batchNormalizerLearningRate = batchNormalizerLearningRate
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
        n.inputs.forEach { (input) in
          input.weight = weights[i][j]
          newNodeWeights.append(input.weight)
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
    self.optimizer = optimizer
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
        weights.append(neuron.inputs.map({ $0.weight }))
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
        let params = normalLobe.normalizerLearningParams
        layer = Layer(activation: lobe.activation,
                      nodes: lobe.neurons.count,
                      weights: weights,
                      type: lobe.layer,
                      bias: biases,
                      biasWeights: biasWeights,
                      normalize: lobe.isNormalized,
                      beta: params.beta,
                      gamma: params.gamma)
      }
      
      layers.append(layer)
    }
    
    let model = ExportModel(layers: layers,
                            learningRate: learningRate,
                            batchNormalizerLearningRate: self.batchNormalizerLearningRate)
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
      lobe = NormalizedLobe(model: model,
                            learningRate: self.learningRate,
                            batchNormLearningRate: self.batchNormalizerLearningRate ?? self.learningRate)
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
                            activation: layer.activation)
        
        neuron.layer = layer.type
        neuron.biasWeight = biasWeight
        
        neurons.append(neuron)
        self.layerWeights.append(weights)
      }
      
      var lobe = Lobe(neurons: neurons,
                      activation: layer.activation)
      
      if layer.normalize {
        lobe = NormalizedLobe(neurons: neurons,
                              activation: layer.activation,
                              beta: layer.beta ?? 0,
                              gamma: layer.gamma ?? 1,
                              learningRate: model.learningRate,
                              batchNormLearningRate: model.batchNormalizerLearningRate ?? model.learningRate)
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
        lobe.layer = layerType
        
        let neuronGroup = self.lobes[i].neurons
        let inputNeuronGroup = self.lobes[i-1].neurons
        
        neuronGroup.forEach { (neuron) in
          neuron.layer = layerType
          
          var dendrites: [NeuroTransmitter] = []
          
          var weights: [Float] = []
          
          for _ in 0..<inputNeuronGroup.count {
            var weight = self.initializer.calculate(m: neuronGroup.count, h: inputNeuronGroup.count)
            
            if let constrain = self.weightConstraints {
              let minBound = constrain.lowerBound
              let maxBound = constrain.upperBound
              weight = min(maxBound, max(minBound, weight))
            }
            
            let transmitter = NeuroTransmitter(weight: weight)
            dendrites.append(transmitter)
            weights.append(weight)
          }
          
          self.layerWeights.append(weights)
          
          let biasWeight = self.initializer.calculate(m: neuronGroup.count, h: inputNeuronGroup.count)
          
          neuron.initializeWeights(count: inputNeuronGroup.count)
          neuron.biasWeight = biasWeight
          neuron.inputs = dendrites
        }
        
      } else {
        
        //first layer weight initialization with 0 since it's just the input layer
        let neuronGroup = self.lobes[i].neurons
        self.lobes[i].layer = .input
        
        for n in 0..<neuronGroup.count {
          
          let transmitter = NeuroTransmitter(weight: 0)
          
          //first layer only has one input per input value
          neuronGroup[n].inputs = [transmitter]
          neuronGroup[n].layer = .input
          
          self.layerWeights.append([0])
        }
        
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
    
    for i in 0..<epochs {
      let mixedData = data
      
      let epochStartDate = Date()
      
      self.log(type: .message, priority: .low, message: "epoch: \(i)")
      
      var percentComplete: Float = 0
      
      switch self.descent {
      case .sgd:
        //train the network with all the data
        for d in 0..<mixedData.count {
          self.zeroGradients()
          
          let obj = data[d]
          
          if d % 20 == 0 {
            percentComplete = Float(d) / Float(data.count)
            self.log(type: .message, priority: .medium, message: "\(percentComplete.percent())%")
          }
          
          self.log(type: .message, priority: .high, message: "data iteration: \(d)")
          
          self.trainIndividual(data: obj)
          
        }
        
      case let .mbgd(size: size):
        let batches = mixedData.batched(into: size)
        
        batches.forEach { (batch) in
          self.zeroGradients()
          
          batch.forEach { (tData) in
            self.trainIndividual(data: tData, backprop: false)
          }
          
          let avgDelta = self.averageDelta()
          let outLayer = self.outputLayer()
          
          for i in 0..<outLayer.count {
            outLayer[i].delta = avgDelta[i]
          }
          
          self.backpropagate()
          self.adjustWeights()
          
          self.descents.removeAll()
        }
      }
      
      //maybe add to serial background queue, dispatch queue crashes
      /// feed a model and its correct values through the network to calculate the loss
      let loss = self.calcAverageLoss(self.feed(input: data[0].data),
                                      correct: data[0].correct)
      self.loss.append(loss)
      
      self.log(type: .message, priority: .low, message: "loss at epoch \(i): \(loss)")
      
      self.log(type: .message,
               priority: .high,
               message: "epoch completed time: \(Date().timeIntervalSince(epochStartDate))")
      
      //feed validation data through the network
      if let validationData = validation.randomElement(), validation.count > 0 {
        
        self.feedInternal(input: validationData.data)
        let errorForValidation = self.calcAverageErrorForOutput(correct: validationData.correct)
        
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
        
        self.log(type: .message, priority: .low, message: "val error at epoch \(i): \(errorForValidation)")
        if previousValidationErrors.count == checkBatchCount {
          previousValidationErrors.removeFirst()
        }
        previousValidationErrors.append(abs(errorForValidation))
        
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
  
  private func averageDelta() -> [Float] {
    var sum: [Float] = [Float].init(repeating: 0, count: self.outputLayer().count)
    
    self.descents.forEach { (descentObj) in
      sum += descentObj
    }
    
    return sum / Float(self.descents.count)
  }
  
  internal func averageError() -> Float {
    var sum: Float = 0
    
    previousValidationErrors.forEach { (error) in
      sum += error
    }
    
    return sum / Float(previousValidationErrors.count)
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
    self.feedInternal(input: input)
    
    let out = self.get()
    return out
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
  
  private func trainIndividual(data: TrainingData, backprop: Bool = true) {
    
    guard data.correct.count == self.outputLayer().count else {
      self.log(type: .error,
               priority: .alwaysShow,
               message: "Error: correct data count does not match ouput node count, bailing out")
      return
    }
    
    //feed the data through the network
    self.feedInternal(input: data.data)
    
    //set the output errors for set
    self.setOutputDeltas(data.correct)
    
    if backprop {
      //propagate backwards through the network
      self.backpropagate()
      
      //adjust all the weights after back propagation is complete
      self.adjustWeights()
    }
  }
  
  /// Feeds the network internally preparing for output or training
  /// - Parameter input: the input data to feed the network
  internal func feedInternal(input: [Float]) {
    var x = input
    
    for i in 0..<self.lobes.count {
      let currentLayer = self.lobes[i]
      let newInputs: [Float] = currentLayer.feed(inputs: x)
      x = newInputs
    }
  }
  
  /// Get the result of the last layer of the network
  /// - Returns: Array of floats resulting from the activation functions of the last layer
  private func get() -> [Float] {
    
    var outputs: [Float] = []
    
    self.outputLayer().forEach { (neuron) in
      outputs.append(neuron.activation())
    }
    
    var out = outputs
    
    if let mod = self.outputModifier {
      var modOut: [Float] = []
      
      for i in 0..<out.count {
        let modVal = mod.calculate(index: i, outputs: out)
        modOut.append(modVal)
      }
      
      out = modOut
    }
    
    return out
  }
  
  /// Get first layer of neurons
  /// - Returns: Input layer as array of Neuron objects
  private func inputLayer() -> [Neuron] {
    return self.lobes.first?.neurons ?? []
  }
  
  /// Get the last layer of neurons
  /// - Returns: Output layer as array of Neuron objects
  private func outputLayer() -> [Neuron] {
    return self.lobes.last?.neurons ?? []
  }
  
  internal func calcAverageLoss(_ predicted: [Float], correct: [Float]) -> Float {
    var sum: Float = 0
    
    for i in 0..<predicted.count {
      let predicted = predicted[i]
      let correct = correct[i]
      let error = self.lossFunction.calculate(predicted, correct: correct)
      sum += error
    }
    
    return sum / Float(predicted.count)
  }
  
  private func calcAverageErrorForOutput(correct: [Float]) -> Float {
    let predicted: [Float] = self.get()
    return self.calcAverageLoss(predicted, correct: correct)
  }
  
  internal func setOutputDeltas(_ correctValues: [Float]) {
    guard correctValues.count == self.outputLayer().count else {
      
      self.log(type: .error,
               priority: .alwaysShow,
               message: "Error: correct data count does not match ouput node count, bailing out")
      return
    }
    
    //set output error delta
    let outs = self.get()
    
    var outputErrors: [Float] = []
    
    for i in 0..<self.outputLayer().count {
      let target = correctValues[i]
      let predicted = outs[i]
      
      let outputNeuron = self.outputLayer()[i]
      
      let delta = self.lossFunction.derivative(predicted, correct: target)
      
      outputNeuron.delta = delta
      outputErrors.append(delta)
      
      self.log(type: .message,
               priority: .high,
               message: "out: \(i), raw: \(outputNeuron.activation()) predicted: \(predicted), actual: \(target) delta: \(String(describing: outputNeuron.delta))")
      
    }
    
    self.descents.append(outputErrors)
  }
  
  internal func gradients() -> [[[Float]]] {
    return self.lobes.map { $0.gradients() }
  }
  
  internal func backpropagate(with deltas: [Float]? = nil) {
    //reverse so we can loop through from the beggining of the array starting at the output node
    let reverse: [Lobe] = self.lobes.reversed()

    guard reverse.count > 0 else {
      return
    }
    
    let outputLayer = reverse[0]
    
//    //for generative adversarial networks we need to set the backprop deltas manually without calculating
//    if let deltas = deltas {
//      outputLayer.setLayerDeltas(with: deltas, update: true)
//    }
//
    var updatingDeltas = deltas ?? outputLayer.deltas()
    
    //subtracting 1 because we dont need to propagate through to the weights in the input layer
    //those will always be 0 since no computation happens at the input layer
    for i in 0..<reverse.count - 1 {
      let currentLobe = reverse[i]
      let previousLobe = reverse[i + 1]
    
      updatingDeltas = currentLobe.backpropagate(inputs: updatingDeltas,
                                                 previousLayerCount: previousLobe.neurons.count)
  
      //incoming inputs are the new deltas for the current layer
      previousLobe.setLayerDeltas(with: updatingDeltas, update: true)
    }
  }
  
  public func zeroGradients() {
    self.lobes.forEach { lobe in
      lobe.zeroGradients()
    }
  }
  
  internal func adjustWeights() {
    for i in 0..<self.lobes.count {
      let lobe = self.lobes[i]
      lobe.adjustWeights(self.weightConstraints)
    }
  }
}
