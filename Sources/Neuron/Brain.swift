//
//  Brain.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/6/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

public class Brain {
  
  /// When set to true this will print the error caclulated by the network
  public var debug: Bool = false
  
  /// The nucleus object that describes the network settings
  public var nucleus: Nucleus
  
  /// Number of times to run the training data set
  public var epochs: Int
  
  /// The loss function data from training to be exported
  private var loss: [Float] = []
  
  /// Neuron matrix in the existing brain object
  private var lobes: [Lobe] = []
  
  /// The function to use to calculate the loss of the network
  private var lossFunction: LossFunction
  
  /// The threshold to compare the validation error with to determine whether or not to stop training.
  /// Default: 0.001. A number between 0 and 0.1 is usually accepted
  private var lossThreshold: Float
  
  /// If the brain object has been compiled and linked properly
  private var compiled: Bool = false
  
  /// Output modifier for the output layer. ex. softmax
  private var outputModifier: OutputModifier? = nil
  
  private var previousValidationError: Float = 99
  
  /// Creates a Brain object that manages a network of Neuron objects
  /// - Parameters:
  ///   - nucleus:  Nucleus object that describes the learning behavior of the network
  ///   - epochs: Number of times to run the training daya
  ///   - lossFunction: The loss function to calculate
  ///   - lossThreshold: The loss threshold to reach when training
  public init(nucleus: Nucleus,
              epochs: Int,
              lossFunction: LossFunction = .meanSquareError,
              lossThreshold: Float = 0.001) {
    
    self.nucleus = nucleus
    self.lossFunction = lossFunction
    self.lossThreshold = lossThreshold
    self.epochs = epochs
  }
  
  /// Adds a layer to the neural network
  /// - Parameter model: The lobe model describing the layer to be added
  public func add(_ model: LobeModel) {
    self.lobes.append(model.lobe(self.nucleus))
  }
  
  /// Adds an output modifier to the output layer
  /// - Parameter mod: The modifier to apply to the outputs
  public func add(modifier mod: OutputModifier) {
    self.outputModifier = mod
  }
  
  /// Connects all the lobes together in the network builing the complete network
  public func compile() {
    guard lobes.count > 0 else {
      fatalError("no lobes to connect bailing out.")
    }
    //link all the layers generating the matrix
    for i in 0..<lobes.count {
      if i > 0 {
        let neuronGroup = self.lobes[i].neurons
        let inputNeuronGroup = self.lobes[i-1].neurons
        
        neuronGroup.forEach { (neuron) in
          let dendrites = [NeuroTransmitter](repeating: NeuroTransmitter(),
                                             count: inputNeuronGroup.count)
          neuron.inputs = dendrites
        }
      }
    }
    
    compiled = true
  }
  
  /// Supervised training function
  /// - Parameters:
  ///   - data: The training data object containing the expected values and the training data
  ///   - validation: The validation data object containing the expected values and the validation data
  ///   - complete: Called when training is completed
  public func train(data: [TrainingData],
                    validation: [TrainingData] = [],
                    complete: ((_ complete: Bool) -> ())? = nil) {
    
    let batchSize = 10

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
      let epochStartDate = Date()
      
      if debug {
        print("epoch: \(i)")
      }
      
      //feed validation data through the network
      if let validationData = validation.randomElement(), validation.count > 0 {
        if debug {
          print("validating....")
        }
        self.feedInternal(input: validationData.data)
        let errorForValidation = self.calcErrorForOutput(correct: validationData.correct)
  
        //if validation error is greater than pervious then we are complete with training
        //bail out to prevent overfitting
        let threshold: Float = self.lossThreshold
        if abs(previousValidationError - errorForValidation) <= threshold {
  
          print("🟢 SUCCESS: training is complete...")
          if debug {
            print("training completed time: \(Date().timeIntervalSince(trainingStartDate))")
          }
          complete?(true)
          return
        }
  
        previousValidationError = errorForValidation
      }
      
      //maybe add to serial background queue, dispatch queue crashes
      /// feed a model and its correct values through the network to calculate the loss
      let loss = self.calcTotalLoss(self.feed(input: data[0].data),
                                    correct: data[0].correct)
      self.loss.append(loss)
      
      
      //train the network with all the data
      for d in 0..<data.count {
        let obj = data[d]
        if debug {
          print("data iteration: \(d)")
        }
        self.trainIndividual(data: obj)
        
        if d % batchSize == 0 {

          //get the error and propagate backwards through
          self.backpropagate(obj.correct)
          //self.backpropagateOptim(correct)
          
          //adjust all the weights after back propagation is complete
          self.adjustWeights()
        }
      }
      
      if debug {
        print("epoch completed time: \(Date().timeIntervalSince(epochStartDate))")
      }
    }
    
    if debug {
      print("training completed time: \(Date().timeIntervalSince(trainingStartDate))")
    }
    complete?(true)
  }
  
  /// Clears the whole network and resets all the weights to a random value
  public func clear() {
    self.previousValidationError = 999
    self.loss.removeAll()
    //clears the whole matrix
    self.lobes.forEach { (lobe) in
      lobe.clear()
    }
  }
  
  /// Feed-forward through the network to get the result
  /// - Parameters:
  ///   - input: the input array of floats
  ///   - ranked: whether the network should sort the output by highest first
  /// - Returns: The result of the feed forward through the network as an array of Floats
  public func feed(input: [Float], ranked: Bool = false) -> [Float] {
    self.feedInternal(input: input)
    
    let out = self.get(ranked: ranked)
    return out
  }
  
  
  /// Update the settings for each lobe and each neuron
  /// - Parameter nucleus: Object that describes the network behavior
  public func updateNucleus(_ nucleus: Nucleus) {
    self.nucleus = nucleus
    
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
  
  private func trainIndividual(data: TrainingData) {
    
    guard data.correct.count == self.outputLayer().count else {
      print("🛑 Error: correct data count does not match ouput node count, bailing out")
      return
    }
    
    //feed the data through the network
    self.feedInternal(input: data.data)
  }
  
  /// Feeds the network internally preparing for output or training
  /// - Parameter input: the input data to feed the network
  private func feedInternal(input: [Float]) {
    self.addInputs(input: input)
    

    for i in 0..<self.lobes.count - 1 {
      let currentLayer = self.lobes[i].neurons


      self.lobes[i + 1].neurons.forEach { (neuron) in

        //THIS IS THE PART THAT TAKES A WHILE!!!
        let newInputs = currentLayer.map { (neuron) -> NeuroTransmitter in
          return NeuroTransmitter(input: neuron.activation())
        }

        neuron.replaceInputs(inputs: newInputs)
      }

    }
//
//    var lastInputs: [NeuroTransmitter]?
//
//    for i in 0..<self.lobes.count {
//      let currentLayer = self.lobes[i].neurons
//
//      lastInputs = currentLayer.map { (neuron) -> NeuroTransmitter in
//        let activated = neuron.activation()
//        return NeuroTransmitter(input: activated)
//      }
//
//      //adjust inputs for next layer
//    }
    

  }
  
  /// Get the result of the last layer of the network
  /// - Parameter ranked: whether the network should sort the output by highest first
  /// - Returns: Array of floats resulting from the activation functions of the last layer
  private func get(ranked: Bool = false) -> [Float] {
    
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
    
    return ranked ? out.sorted(by: { $0 > $1 }) : out
  }
  
  /// Adds inputs to the input layer where the NeuroTransmitter links a value
  /// - Parameter input: Input array of floats
  private func addInputs(input: [Float]) {
    guard input.count == self.inputLayer().count else {
      print("🛑 Error: input data count does not match input node count, bailing out")
      return
    }
    
    for i in 0..<inputLayer().count {
      let inputNode = inputLayer()[i]
      let inputValue = input[i]
      
      //one input per input node
      inputNode.addInput(input: NeuroTransmitter(input: inputValue), at: 0)
    }
    
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
  
  private func calcTotalLoss(_ predicted: [Float], correct: [Float]) -> Float {
    let error = self.lossFunction.calculate(predicted, correct: correct)
    return error
  }
  
  private func calcErrorForOutput(correct: [Float]) -> Float {
    let predicted: [Float] = self.outputLayer().map({ $0.activation() })
    return self.calcTotalLoss(predicted, correct: correct)
  }
  
  private func backpropagate(_ correctValues: [Float]) {
    guard correctValues.count == self.outputLayer().count else {
      print("🛑 Error: correct data count does not match ouput node count, bailing out")
      return
    }
    
    let outputs = self.outputLayer()
    //set output error delta
    for i in 0..<correctValues.count {
      
      let correct = correctValues[i]
      let outputNeuron = outputs[i]
      let get = outputNeuron.activation()
      outputs[i].delta = correct - get
      if debug {
        print("out: \(i), predicted: \(get), actual: \(correct)")
      }
    }
    
    //reverse so we can loop through from the beggining of the array starting at the output node
    let reverse: [Lobe] = self.lobes.reversed()
    
    //- 1 because we dont need to propagate passed the input layer
    //DISPATCH QUEUE BREAKS EVERYTHING NEED BETTER OPTIMIZATION =(
    //WITH OUT IT MAKES IT MUCH SLOWER BUT WITH IT IT FORMS A RACE CONDITION =(
    
    for i in 0..<reverse.count - 1 {
      let currentLayer = reverse[i].neurons
      let previousLayer = reverse[i + 1].neurons
      
      for p in 0..<previousLayer.count {
        var deltaAtLayer: Float = 0
        
        for c in 0..<currentLayer.count {
          let currentNode = currentLayer[c]
          
          let currentNeuronDelta = currentNode.delta * currentNode.inputs[p].weight
          deltaAtLayer += currentNeuronDelta
        }
        
        previousLayer[p].delta = deltaAtLayer

      }
      
    }
    
  }
  
  private func adjustWeights() {
    for i in 0..<self.lobes.count {
      let lobe = self.lobes[i]
      lobe.adjustWeights()
    }
  }
}
