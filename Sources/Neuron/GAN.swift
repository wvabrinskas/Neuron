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
  private var generatorEpochs: Int
  
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
              generatorEpochs: Int,
              lossThreshold: Float = 0.001,
              initializer: Initializers = .xavierNormal,
              descent: GradientDescent = .sgd) {
    
    self.epochs = epochs
    self.generatorEpochs = generatorEpochs
    //generator
    let brainGen = Brain(learningRate: learningRate,
                         epochs: generatorEpochs,
                         lossFunction: .crossEntropy,
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
                         lossFunction: .crossEntropy,
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
    brainDis.add(modifier: .softmax)
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

  private func trainGenerator(_ complete: ((_ success: Bool) -> ())? = nil) {
    //input random data to generator
    //get generator output
    //feed that to the discriminator
    //get the error at the output
    //feed that back through the discriminator
    //get the delatas at hidden
    //feed those deltas to the generator
    //adjust weights of generator
    self.discriminator.loss.removeAll()
    
    for i in 0..<self.generatorEpochs {
      let sample = self.getGeneratedSample()
      
      //feed sample
      let output = self.discriminate(sample)
      
      //calculate loss at discrimator
      let loss = self.discriminator.calcAverageLoss(output, correct: [1.0, 0.0])
      
      self.discriminator.loss.append(loss)
      self.discriminator.log(type: .message, priority: .low, message: "loss at epoch \(i): \(loss)")
      
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
      //repeat
    }
    
    self.generator.log(type: .success,
             priority: .alwaysShow,
             message: "Generator training completed")
    
    self.generator.log(type: .message, priority: .alwaysShow, message: "Loss: \(self.discriminator.loss.last ?? 0)")
    
    complete?(true)
  }
  
  private func trainDiscriminator(data: [TrainingData],
                                  singleStep: Bool = false,
                                  complete: ((_ complete: Bool) -> ())? = nil) {
    guard data.count > 0 else {
      return
    }
    //input real data to discrimator
    //get the classifier output
    //calc the error
    //backprop regular through the discriminator
    //adjust weights
    //do we just train once get result? Or train multiple times off the real data?
    self.discriminator.epochs = singleStep ? 1 : self.epochs
    self.discriminator.train(data: data, complete: complete)
  }
  
  public func train(type: GANTrainingType,
                    data: [TrainingData] = [],
                    singleStep: Bool = false,
                    complete: ((_ success: Bool) -> ())? = nil) {
    switch type {
    case .discriminator:
      print("training discriminator")
      self.trainDiscriminator(data: data, singleStep: singleStep, complete: complete)
    case .generator:
      print("training generator")
      self.trainGenerator(complete)
    }
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
