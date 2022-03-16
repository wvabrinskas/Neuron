//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/21/21.
//

import Foundation
import Logger
import NumSwift


typealias Output = (loss: Float, output: [Float])


public class GAN: Logger, GANTrainingDataBuilder, GANDefinition, Trainable {
  public typealias TrainableDatasetType = TrainingData
  
  internal var generator: Brain?
  internal var discriminator: Brain?
  internal var batchSize: Int
  internal var gradientPenaltyLambda: Float = 10
  internal var criticTrainPerEpoch: Int = 5
  internal var discriminatorLossHistory: [Float] = []
  internal var totalCorrectGuesses: Int = 0
  internal var totalGuesses: Int = 0
  
  public var generatorLossHistory: [Float] = []
  public var gradientPenaltyHistory: [Float] = []
  public var lossFunction: GANLossFunction = .minimax
  public var epochs: Int
  public var logLevel: LogLevel = .none
  public var randomNoise: () -> [Float]
  public var validateGenerator: (_ output: [Float]) -> Bool
  public var discriminatorNoiseFactor: Float?
  public var metricsToGather: Set<Metric> = []
  public var metrics: [Metric : Float] = [:]
  
  @TestNaN public var discriminatorLoss: Float = 0 {
    didSet {
      self.discriminatorLossHistory.append(discriminatorLoss)
      addMetric(value: discriminatorLoss, key: .criticLoss)
    }
  }
  
  @TestNaN public var generatorLoss: Float = 0 {
    didSet {
      self.generatorLossHistory.append(generatorLoss)
      addMetric(value: generatorLoss, key: .generatorLoss)
    }
  }
  
  @TestNaN public var gradientPenalty: Float = 0 {
    didSet {
      self.gradientPenaltyHistory.append(gradientPenalty)
      addMetric(value: gradientPenalty, key: .gradientPenalty)
    }
  }
  
  //MARK: Init
  public init(generator: Brain? = nil,
              discriminator: Brain? = nil,
              epochs: Int,
              criticTrainPerEpoch: Int = 5,
              gradientPenaltyLambda: Float = 10,
              batchSize: Int,
              metrics: Set<Metric> = []) {
    
    self.epochs = epochs
    self.batchSize = batchSize
    self.criticTrainPerEpoch = criticTrainPerEpoch
    self.generator = generator
    self.discriminator = discriminator
    self.gradientPenaltyLambda = gradientPenaltyLambda
    self.metricsToGather = metrics
    
    self.randomNoise = {
      var noise: [Float] = []
      for _ in 0..<10 {
        noise.append(Float.random(in: 0...1))
      }
      return noise
    }
    
    self.validateGenerator = { output in
      return false
    }
  }
  
  public func trainOn(_ batch: [TrainingData]) -> Float {
    //no op
    return 0
  }
  
  public func validateOn(_ batch: [TrainingData]) -> Float {
    //no op
    return 0
  }
  
  private func checkGeneratorValidation(for epoch: Int) -> Bool {
    if epoch % 5 == 0 {
      return self.validateGenerator(getGeneratedSample())
    }
    
    return false
  }
  
  private func startTraining(dataset: InputData,
                             epochCompleted: ((Int, [Metric : Float]) -> ())? = nil,
                             complete: (([Metric : Float]) -> ())? = nil) {
    
    guard let dis = self.discriminator, let gen = self.generator else {
      return
    }
    
    let data = dataset.training
    
    guard data.count > 0 else {
      return
    }
    
    let epochs = self.epochs
    
    self.log(type: .message, priority: .alwaysShow, message: "Training started...")
    
    for i in 0..<epochs {
      //prepare data into batches
      for _ in 0..<self.criticTrainPerEpoch {
        dis.zeroGradients()
        
        //freeze the generator
        gen.trainable = false
        dis.trainable = true
        
        let noise = randomNoise()
        
        let realDataBatch = self.getRandomBatch(data: data)
        let fakeDataBatch = self.getGeneratedData(type: .fake, noise: noise)
        
        self.discriminatorLoss = criticStep(real: realDataBatch, fake: fakeDataBatch)
        
        dis.backpropagate(with: [discriminatorLoss])
        dis.adjustWeights(batchSize: 1)
      }
      
      //freeze the generator
      gen.trainable = true
      dis.trainable = false
      
      let noise = randomNoise()
      //zero out gradients before training generator
      gen.zeroGradients()
      dis.zeroGradients()
      
      //train generator on newly trained discriminator
      let generatedData = self.getGeneratedData(type: .real, noise: noise)
      
      generatorLoss = generatorStep(fake: generatedData)
      
      //backprop discrimator
      let firstLayerDeltas = dis.backpropagate(with: [self.generatorLoss]).firstLayerDeltas
      
      //get discriminator gradients for each generator parameter first
      gen.backpropagate(with: firstLayerDeltas)
      gen.adjustWeights(batchSize: 1) //figure out batch size?
      
      epochCompleted?(i, metrics)
      
      if self.checkGeneratorValidation(for: i) {
        return
      }
    }
    
    self.log(type: .message, priority: .alwaysShow, message: "GAN Training complete")
    
    complete?(metrics)
  }
  
  /// Discrimate on a batch and return the average loss on the batch with each output
  /// - Parameters:
  ///   - batch: batch of TrainingData to discriminate against
  ///   - type: Type of data being passed in
  /// - Returns: The average loss on the batch and an array of outputs from each data point
  internal func batchDiscriminate(_ batch: [TrainingData],
                                  type: GANTrainingType) -> Output {
    //train on each sample
    var outputs: [Float] = []
    
    var averageLoss: Float = 0
    
    for i in 0..<batch.count {
      //get sample from generator
      let sample = batch[i].data
      
      //feed sample
      let output = discriminate(sample)
      
      //append outputs for wasserstein
      outputs.append(output)
      
      //get loss for type
      let loss = self.lossFunction.loss(type, value: output)
      
      //add losses together
      averageLoss += loss / Float(batch.count)
    }
    
    //get average loss over batch
    return (loss: averageLoss, output: outputs)
  }
  
  internal func generatorStep(fake: [TrainingData]) -> Float {
    let genOutput = self.batchDiscriminate(fake, type: .real)
    return -1 * genOutput.loss
  }
  
  internal func criticStep(real: [TrainingData],
                           fake: [TrainingData]) -> Float {
    
    //train discriminator on real data
    let realOutput = batchDiscriminate(real, type: .real)
    
    //tran discriminator on new fake data
    let fakeOutput = batchDiscriminate(fake, type: .fake)
    
    let realLoss = realOutput.loss
    let fakeLoss = fakeOutput.loss
    
    //adding real and fake based on minimax loss function of log(D(x)) + log(D(G(z)))
    let totalSumLoss = realLoss + fakeLoss
    //we are taking the sum of all instances of the minibatch and dividing by batch size
    //to get average loss
    //negative because the Neuron only supports MINIMIZING gradients
    let averageTotalLoss = -1 * totalSumLoss
    
    return averageTotalLoss
  }
  
  //MARK: Public Functions
  public func add(generator gen: Brain) {
    self.generator = gen
    gen.compile()
  }
  
  public func add(discriminator dis: Brain) {
    guard dis.lobes.last?.neurons.count == 1 else {
      self.log(type: .error, priority: .alwaysShow, message: "Discriminator should only have 1 output neuron")
      return
    }
    self.discriminator = dis
    dis.compile()
  }
  
  public func train(dataset: InputData,
                    epochCompleted: ((Int, [Metric : Float]) -> ())? = nil,
                    complete: (([Metric : Float]) -> ())? = nil) {
    
    self.startTraining(dataset: dataset,
                       epochCompleted: epochCompleted,
                       complete: complete)
  }
  
  @discardableResult
  public func discriminate(_ input: [Float]) -> Float {
    guard let dis = self.discriminator else {
      return 0
    }
    //we only expect ONE output neuron
    let output = dis.feed(input: input).first ?? 0
    return output
  }
  
  public func getGeneratedSample() -> [Float] {
    guard let gen = self.generator else {
      return []
    }
    
    return gen.feed(input: randomNoise())
  }
  
}
