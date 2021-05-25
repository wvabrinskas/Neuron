//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/21/21.
//

import Foundation
import Logger

public protocol GANModel {
  var hiddenLayers: Int { get set }
  var hiddenNodesPerLayer: Int { get set }
  var bias: Float { get set }
  var activation: Activation { get set }
  var learningRate: Float { get set }
}

public struct GeneratorModel: GANModel {
  public var hiddenLayers: Int
  public var hiddenNodesPerLayer: Int
  public var bias: Float
  public var activation: Activation
  public var inputs: Int
  public var outputs: Int
  public var learningRate: Float
  
  public init(inputs: Int,
              hiddenLayers: Int,
              hiddenNodesPerLayer: Int,
              outputs: Int,
              bias: Float,
              activation: Activation = .tanh,
              learningRate: Float = 0.001) {
    self.inputs = inputs
    self.outputs = outputs
    self.hiddenLayers = hiddenLayers
    self.hiddenNodesPerLayer = hiddenNodesPerLayer
    self.bias = bias
    self.activation = activation
    self.learningRate = learningRate
  }
}

public struct DiscriminatorModel: GANModel {
  public var hiddenLayers: Int
  public var hiddenNodesPerLayer: Int
  public var bias: Float
  public var activation: Activation
  public var learningRate: Float

  public init(hiddenLayers: Int,
              hiddenNodesPerLayer: Int,
              bias: Float,
              activation: Activation = .sigmoid,
              learningRate: Float = 0.001) {
    self.hiddenLayers = hiddenLayers
    self.hiddenNodesPerLayer = hiddenNodesPerLayer
    self.bias = bias
    self.activation = activation
    self.learningRate = learningRate
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
  
  public init(discriminatorModel: DiscriminatorModel,
              generatorModel: GeneratorModel,
              epochs: Int,
              lossThreshold: Float = 0.001,
              initializer: Initializers = .xavierNormal,
              descent: GradientDescent = .sgd,
              batchSize: Int) {
    
    self.epochs = epochs
    self.batchSize = batchSize
    self.lossTreshold = lossThreshold
    
    //generator
    let brainGen = Brain(learningRate: generatorModel.learningRate,
                         epochs: epochs,
                         lossFunction: .binaryCrossEntropy,
                         lossThreshold: lossThreshold,
                         initializer: initializer,
                         descent: descent)
    
    brainGen.add(LobeModel(nodes: generatorModel.inputs))
    
    for _ in 0..<generatorModel.hiddenLayers {
      brainGen.add(LobeModel(nodes: generatorModel.hiddenNodesPerLayer,
                             activation: .leakyRelu,
                             bias: generatorModel.bias))
    }
    
    brainGen.add(LobeModel(nodes: generatorModel.outputs,
                           activation: generatorModel.activation,
                           bias: generatorModel.bias))
    
    brainGen.add(optimizer: .adam())
    brainGen.logLevel = self.logLevel
    
    self.generator = brainGen
    self.generator.compile()
    
    //discriminator
    let brainDis = Brain(learningRate: discriminatorModel.learningRate,
                         epochs: epochs,
                         lossFunction: .binaryCrossEntropy,
                         lossThreshold: lossThreshold,
                         initializer: initializer,
                         descent: descent)
    
    //inputs of discrimnator should be the same as outputs of the generator
    brainDis.add(LobeModel(nodes: generatorModel.outputs)) //input
    
    for _ in 0..<discriminatorModel.hiddenLayers {
      brainDis.add(LobeModel(nodes: discriminatorModel.hiddenNodesPerLayer,
                             activation: .leakyRelu,
                             bias: discriminatorModel.bias))
    }
    
    brainDis.add(LobeModel(nodes: 2,
                           activation: discriminatorModel.activation,
                           bias: discriminatorModel.bias)) //output class count is 2 because "real or fake" is two classes
    
    brainDis.add(optimizer: .adam())
    brainDis.logLevel = self.logLevel

    self.discriminator = brainDis
    self.discriminator.compile()
    
    self.randomNoise = {
      var noise: [Float] = []
      for _ in 0..<generatorModel.inputs {
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
      let output = self.discriminate(sample)
      
      //calculate loss at discrimator
      let loss = self.discriminator.calcAverageLoss(output, correct: [1.0, 0.0])
      self.generator.loss.append(loss)
      
      self.generator.log(type: .message, priority: .alwaysShow, message: "Generator loss: \(loss)")
      
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

  private func getFakeData(_ data: [TrainingData]) -> [TrainingData] {
    var fakeData: [TrainingData] = []
    for _ in 0..<data.count {
      let sample = self.getGeneratedSample()
      let training = TrainingData(data: sample, correct: [0.0, 1.0])
      fakeData.append(training)
    }
    
    return fakeData
  }
  
  private func startTraining(data: [TrainingData],
                             singleStep: Bool = false,
                             complete: ((_ complete: Bool) -> ())? = nil) {
    
    guard data.count > 0 else {
      return
    }

    let fakeData = self.getFakeData(data)
    //mix fake into real
    var realDataMixedWithFake = data
    realDataMixedWithFake.append(contentsOf: fakeData)

    //prepare data into batches
    var realFakeBatched = self.getBatchedRandomData(data: realDataMixedWithFake)
    let epochs = singleStep ? 1 : self.epochs
    
    //control epochs locally
    self.discriminator.epochs = 1
    
    //train on real
    for i in 0..<epochs {
      if self.checkGeneratorValidation(for: i) {
        return
      }
      //train discriminator
      print("training discriminator....")
      let randomRealFakeIndex = Int.random(in: 0..<realFakeBatched.count)
      let newRealFakeBatch = realFakeBatched[randomRealFakeIndex]
      
      //train discriminator on real data combined with fake data
      self.discriminator.train(data: newRealFakeBatch)
      
      //train generator on newly trained discriminator
      self.trainGenerator()
      
      //update fake data
      let fakeData = self.getFakeData(data)

      var newData = data
      newData.append(contentsOf: fakeData)

      //prepare data into batches
      realFakeBatched = self.getBatchedRandomData(data: realDataMixedWithFake)
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
