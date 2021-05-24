//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/21/21.
//

import Foundation
import Logger

public struct GANModel {
  public var inputs: Int
  public var hiddenLayers: Int
  public var hiddenNodesPerLayer: Int
  public var outputs: Int
  public var generatorInputs: Int
  public var bias: Float
  
  public init(inputs: Int,
              hiddenLayers: Int,
              hiddenNodesPerLayer: Int,
              outputs: Int,
              generatorInputs: Int,
              bias: Float) {
    self.inputs = inputs
    self.hiddenLayers = hiddenLayers
    self.hiddenNodesPerLayer = hiddenNodesPerLayer
    self.outputs = outputs
    self.generatorInputs = generatorInputs
    self.bias = bias
  }
}


public class GAN {
  private var generator: Brain
  private var discriminator: Brain
  private var epochs: Int
  private var batchSize: Int
  private var lossTreshold: Float
  
  public var logLevel: LogLevel = .none
  
  public enum GANTrainingType {
    case discriminator, generator
  }
  
  public var randomNoise: () -> [Float]
  
  //create two networks
  //one for the generator
  //one for the discriminator
  // link them using neuron network
  //backprop can skip input layer of discriminator when trainign generator
  //backprop will of generator is as saame as a regular NN
  
  public init(ganModel: GANModel,
              learningRate: Float,
              epochs: Int,
              lossThreshold: Float = 0.001,
              initializer: Initializers = .xavierNormal,
              descent: GradientDescent = .sgd,
              batchSize: Int) {
    
    self.epochs = epochs
    self.batchSize = batchSize
    self.lossTreshold = lossThreshold
    
    //generator
    let brainGen = Brain(learningRate: learningRate,
                         epochs: epochs,
                         lossFunction: .binaryCrossEntropy,
                         lossThreshold: lossThreshold,
                         initializer: initializer,
                         descent: descent)
    
    brainGen.add(LobeModel(nodes: ganModel.generatorInputs))
    
    for _ in 0..<ganModel.hiddenLayers {
      brainGen.add(LobeModel(nodes: ganModel.hiddenNodesPerLayer,
                             activation: .reLu,
                             bias: ganModel.bias))
    }
    
    brainGen.add(LobeModel(nodes: ganModel.outputs, activation: .tanh, bias: ganModel.bias))
    brainGen.add(optimizer: .adam())
    brainGen.logLevel = self.logLevel
    
    self.generator = brainGen
    self.generator.compile()
    
    //discriminator
    let brainDis = Brain(learningRate: learningRate,
                         epochs: epochs,
                         lossFunction: .binaryCrossEntropy,
                         lossThreshold: lossThreshold,
                         initializer: initializer,
                         descent: descent)
    
    //inputs of discrimnator should be the same as outputs of the generator
    brainDis.add(LobeModel(nodes: ganModel.outputs)) //input
    for _ in 0..<ganModel.hiddenLayers {
      brainDis.add(LobeModel(nodes: ganModel.hiddenNodesPerLayer,
                             activation: .leakyRelu,
                             bias: ganModel.bias))
    }
    brainDis.add(LobeModel(nodes: 2, activation: .sigmoid, bias: ganModel.bias)) //output class count is 2 because "real or fake" is two classes
    
    //discriminator has softmax output
  //  brainDis.add(modifier: .softmax)
    brainDis.add(optimizer: .adam())
    brainDis.logLevel = self.logLevel

    self.discriminator = brainDis
    self.discriminator.compile()
    
    self.randomNoise = {
      var noise: [Float] = []
      for _ in 0..<ganModel.generatorInputs {
        noise.append(Float.random(in: 0...1))
      }
      return noise
    }
  }

  //single step operation only
  private func trainGenerator(_ count: Int) {
    //input random data to generator
    //get generator output
    //feed that to the discriminator
    //get the error at the output
    //feed that back through the discriminator
    //get the delatas at hidden
    //feed those deltas to the generator
    //adjust weights of generator
    
    //[Noise] -> [Generator] -> [Sample] -> [Discriminator] -> [real] / [fake]
    //<----------[adjust_w]-----------------[backprop]-----
    
    //train on each sample
    for _ in 0..<count {
      let sample = self.getGeneratedSample()
      //feed sample
      let output = self.discriminate(sample)
      
      //calculate loss at discrimator
      let loss = self.generator.calcAverageLoss(output, correct: [1.0, 0.0])
      self.generator.loss.append(loss)
      
      let lossDiscriminator = self.discriminator.calcAverageLoss(output, correct: [1.0, 0.0])
      self.discriminator.loss.append(lossDiscriminator)
      
      //calculate loss at last layer for discrimator
      //we want it to be real so correct is [1.0, 0.0] [real, fake]
      let trainingData = TrainingData(data: sample, correct: [1.0, 0.0])
      self.discriminator.setOutputDeltas(trainingData.correct)
      
      //we might need ot manage the training ourselves because of the whole not wanting to adjust weights thing
      //and we need to pass the backprop to generator from discriminator
       
      //backprop discrimator
      self.discriminator.backpropagate()
      
      //get deltas from discrimator
      if let deltas = self.discriminator.lobes.first(where: { $0.deltas().count > 0 })?.deltas() {
        
        self.generator.backpropagate(with: deltas)
        
        //adjust weights of generator
        self.generator.adjustWeights()
      }
    }
    //repeat
    
    self.generator.log(type: .success,
             priority: .alwaysShow,
             message: "Generator training completed")
    
    self.generator.log(type: .message, priority: .alwaysShow, message: "Loss: \(self.generator.loss.last ?? 0)")
    
  }
  
  private func startTraining(data: [TrainingData],
                             validation: [TrainingData] = [],
                             singleStep: Bool = false,
                             complete: ((_ complete: Bool) -> ())? = nil) {
    
    guard data.count > 0 else {
      return
    }
    
    var fakeData: [TrainingData] = []
    var fakeValidationData: [TrainingData] = []
    
    for _ in 0..<data.count {
      let sample = self.getGeneratedSample()
      let training = TrainingData(data: sample, correct: [0.0, 1.0])
      let validationSample = self.getGeneratedSample()
      let trainingValidation = TrainingData(data: validationSample, correct: [0.0, 1.0])
      fakeValidationData.append(trainingValidation)
      fakeData.append(training)
    }
    
    var trainingData = data
    trainingData.append(contentsOf: fakeData)
    let randomData = trainingData.randomize()
    
    var validationData = validation
    validationData.append(contentsOf: fakeValidationData)
    let randomValidationData = validationData.randomize()

    let batchedData = randomData.batched(into: self.batchSize)
    let randomBatchedIndex = Int.random(in: 0..<batchedData.count)
    let randomBatch = batchedData[randomBatchedIndex]
    
    let batchedValidationData = randomValidationData.batched(into: self.batchSize)
    let randomBatchValidationIndex = Int.random(in: 0..<batchedValidationData.count)
    let randomValidationBatch = batchedValidationData[randomBatchValidationIndex]
    
    let epochs = singleStep ? 1 : self.epochs
    
    //create a loop that sets the epochs to 1 until self.epochs is empty
    //each iteration we train the generator for 1 epoch until self.generatorEpochs is empty
    
    for _ in 0..<epochs {
      if self.discriminator.averageError() <= self.lossTreshold {
        complete?(true)
        return
      }
      
      self.discriminator.epochs = 1
      
      //train discriminator
      print("training discriminator....")
      self.discriminator.train(data: randomBatch, validation: randomValidationBatch) { success in
        //train generator
        print("training generator....")
        self.trainGenerator(randomBatch.count)
      }
    }
    
    complete?(false)
  }
  
  
  
  public func train(data: [TrainingData] = [],
                    validation: [TrainingData] = [],
                    singleStep: Bool = false,
                    complete: ((_ success: Bool) -> ())? = nil) {
    
    self.startTraining(data: data,
                       validation: validation,
                       singleStep: singleStep,
                       complete: complete)
  }
  
  @discardableResult
  public func discriminate(_ input: [Float]) -> [Float] {
    let output = self.discriminator.feed(input: input)
    return output
  }
  
  public func getGeneratedSample() -> [Float] {
    //need to feed generator random noise
    //return generator output after training
    //probably will be the size of the output
    //add function that will generator random noise
    return generator.feed(input: randomNoise())
  }
  
}
