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
  public var generatorOutputActivation: Activation
  public var discriminatorOutputActivation: Activation
  
  public init(inputs: Int,
              hiddenLayers: Int,
              hiddenNodesPerLayer: Int,
              outputs: Int,
              generatorInputs: Int,
              bias: Float,
              generatorOutputActivation: Activation = .tanh,
              discriminatorOutputActivation: Activation = .sigmoid) {
    self.inputs = inputs
    self.hiddenLayers = hiddenLayers
    self.hiddenNodesPerLayer = hiddenNodesPerLayer
    self.outputs = outputs
    self.generatorInputs = generatorInputs
    self.bias = bias
    self.generatorOutputActivation = generatorOutputActivation
    self.discriminatorOutputActivation = discriminatorOutputActivation
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
  public var validateGenerator: (_ output: [Float]) -> Bool
  
  //create two networks
  //one for the generator
  //one for the discriminator
  // link them using neuron network
  //backprop can skip input layer of discriminator when trainign generator
  //backprop will of generator is as saame as a regular NN
  
  public init(ganModel: GANModel,
              learningRate: Float,
              generatorLearningRate: Float,
              epochs: Int,
              lossThreshold: Float = 0.001,
              initializer: Initializers = .xavierNormal,
              descent: GradientDescent = .sgd,
              batchSize: Int) {
    
    self.epochs = epochs
    self.batchSize = batchSize
    self.lossTreshold = lossThreshold
    
    //generator
    let brainGen = Brain(learningRate: generatorLearningRate,
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
    
    brainGen.add(LobeModel(nodes: ganModel.outputs,
                           activation: ganModel.generatorOutputActivation))
    
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
    
    brainDis.add(LobeModel(nodes: 2,
                           activation: ganModel.discriminatorOutputActivation)) //output class count is 2 because "real or fake" is two classes
    
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
    
    self.validateGenerator = { output in
      return false
    }
  }

  //single step operation only
  private func trainGenerator() {
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
    for _ in 0..<self.batchSize {
      //get sample from generator
      let sample = self.getGeneratedSample()
      let trainingData = TrainingData(data: sample, correct: [1.0, 0.0])

      //feed sample
      self.discriminator.feedInternal(input: trainingData.data)
      
//      //calculate loss at discrimator
//      let loss = self.generator.calcAverageLoss(output, correct: [1.0, 0.0])
//      self.generator.loss.append(loss)
      
      //calculate loss at last layer for discrimator
      //we want it to be real so correct is [1.0, 0.0] [real, fake]
      self.discriminator.setOutputDeltas(trainingData.correct)
      
      //we might need ot manage the training ourselves because of the whole not wanting to adjust weights thing
      //and we need to pass the backprop to generator from discriminator
       
      //backprop discrimator
      self.discriminator.backpropagate()
      
      //get deltas from discrimator
      if self.discriminator.lobes.count > 1 {
        let deltas = self.discriminator.lobes[1].deltas()
        self.generator.backpropagate(with: deltas)
        
        //adjust weights of generator
        self.generator.adjustWeights()
      }
    }
    //repeat
    
  }
  
  private func getBatchedRandomData(data: [TrainingData]) -> [[TrainingData]] {
    let random = data.randomize()
    let preBatched = random.batched(into: self.batchSize)
    return preBatched
  }
  
  private func checkGeneratorValidation(for epoch: Int) -> Bool {
    if epoch % 5 == 0 {
      return self.validateGenerator(getGeneratedSample())
    }
    
    return false
  }

  private func startTraining(data: [TrainingData],
                             singleStep: Bool = false,
                             complete: ((_ complete: Bool) -> ())? = nil) {
    
    guard data.count > 0 else {
      return
    }

    var fakeData: [TrainingData] = []

    //create fake data
    for _ in 0..<data.count {
      let sample = self.getGeneratedSample()
      let training = TrainingData(data: sample, correct: [0.0, 1.0])
      fakeData.append(training)
    }

    //prepare data into batches
    let fakeBatched = self.getBatchedRandomData(data: fakeData)
    let realBatched = self.getBatchedRandomData(data: data)
    
    let epochs = singleStep ? 1 : self.epochs
    
    //control epochs locally
    self.discriminator.epochs = 1
    
    //train on real
    for i in 0..<epochs {
      if self.checkGeneratorValidation(for: i) {
        return
      }
      //train discriminator
      print("training discriminator on real....")
      //get random batch
      let randomRealIndex = Int.random(in: 0..<realBatched.count)
      let newRealBatch = realBatched[randomRealIndex]
      
      self.discriminator.train(data: newRealBatch)
    }
    
    //train on fake
    for i in 0..<epochs {
      if self.checkGeneratorValidation(for: i) {
        return
      }
      //train discriminator
      print("training discriminator on fake....")
      //get random batch
      let randomFakeBatchedIndex = Int.random(in: 0..<fakeBatched.count)
      let newFakeBatch = fakeBatched[randomFakeBatchedIndex]
            
      self.discriminator.train(data: newFakeBatch)
    }
    
    print("training generator....")
    //train generator on discriminator
    for i in 0..<epochs {
      if self.checkGeneratorValidation(for: i) {
        return
      }
      //train generator
      self.trainGenerator()
    }
    
    print("complete")
    
    complete?(false)
  }
  
  
  
  public func train(data: [TrainingData] = [],
                    singleStep: Bool = false,
                    complete: ((_ success: Bool) -> ())? = nil) {
    
    self.startTraining(data: data,
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
