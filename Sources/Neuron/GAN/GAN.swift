//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/21/21.
//

import Foundation
import Logger
import NumSwift


public class GAN: Logger, GANTrainingDataBuilder {
  internal var generator: Brain?
  internal var discriminator: Brain?
  internal var batchSize: Int
  
  private var criticTrainPerEpoch: Int = 5
  private var discriminatorLossHistory: [Float] = []
  private var generatorLossHistory: [Float] = []
  private var gradientPenaltyHistory: [Float] = []
  private var gradientPenaltyCenter: Float = 1
  private var gradientPenaltyLambda: Float = 10
  
  public var epochs: Int
  public var logLevel: LogLevel = .none
  public var randomNoise: () -> [Float]
  public var validateGenerator: (_ output: [Float]) -> Bool
  public var discriminatorNoiseFactor: Float?
  public var lossFunction: GANLossFunction = .minimax
  
  @TestNaN public var discriminatorLoss: Float = 0 {
    didSet {
      self.discriminatorLossHistory.append(discriminatorLoss)
    }
  }
  
  @TestNaN public var generatorLoss: Float = 0 {
    didSet {
      self.generatorLossHistory.append(generatorLoss)
    }
  }
  
  @TestNaN public var gradientPenalty: Float = 0 {
    didSet {
      self.gradientPenaltyHistory.append(gradientPenalty)
    }
  }

  //MARK: Init
  public init(generator: Brain? = nil,
              discriminator: Brain? = nil,
              epochs: Int,
              criticTrainPerEpoch: Int = 5,
              gradientPenaltyCenter: Float = 0,
              gradientPenaltyLambda: Float = 10,
              batchSize: Int) {
    
    self.epochs = epochs
    self.batchSize = batchSize
    self.criticTrainPerEpoch = criticTrainPerEpoch
    self.generator = generator
    self.discriminator = discriminator
    self.gradientPenaltyCenter = gradientPenaltyCenter
    self.gradientPenaltyLambda = gradientPenaltyLambda
    
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

  private func checkGeneratorValidation(for epoch: Int) -> Bool {
    if epoch % 5 == 0 {
      return self.validateGenerator(getGeneratedSample())
    }
    
    return false
  }

  private func startTraining(data: [TrainingData],
                             epochCompleted: ((_ epoch: Int) -> ())? = nil,
                             complete: ((_ complete: Bool) -> ())? = nil) {
    
    guard let dis = self.discriminator, let gen = self.generator else {
      return
    }

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
        let fakeDataBatch = self.getGeneratedData(type: .fake,
                                                  noise: noise,
                                                  size: self.batchSize)
                
        if self.lossFunction == .minimax {
          let minimaxLoss = minimaxDiscriminator(real: realDataBatch, fake: fakeDataBatch)
          self.discriminatorLoss = minimaxLoss

        } else if self.lossFunction == .wasserstein {
          let criticLoss = wassersteinCritic(real: realDataBatch, fake: fakeDataBatch, withPenalty: true)
          self.discriminatorLoss = criticLoss
        }
        
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
      let generatedData = self.getGeneratedData(type: .real,
                                                noise: noise,
                                                size: self.batchSize)
      
      if self.lossFunction == .minimax {
        self.generatorLoss =  minimaxGenerator(fake: generatedData)
        
      } else if self.lossFunction == .wasserstein {
        self.generatorLoss = wassersteinGenerator(fake: generatedData)
      }
      
      //backprop discrimator
      let firstLayerDeltas = dis.backpropagate(with: [self.generatorLoss]).firstLayerDeltas
              
      //get discriminator gradients for each generator parameter first
      gen.backpropagate(with: firstLayerDeltas)
      gen.adjustWeights(batchSize: 1) //figure out batch size?
      
      epochCompleted?(i)
      
      if self.checkGeneratorValidation(for: i) {
        return
      }
    }
    
    self.log(type: .message, priority: .alwaysShow, message: "GAN Training complete")

    complete?(false)
  }
  
  
  /// Discrimate on a batch and return the average loss on the batch with each output
  /// - Parameters:
  ///   - batch: batch of TrainingData to discriminate against
  ///   - type: Type of data being passed in
  /// - Returns: The average loss on the batch and an array of outputs from each data point
  private func disriminateOn(batch: [TrainingData],
                             type: GANTrainingType) -> (loss: Float, output: [Float]) {
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
  
//MARK: Public Functions
  public func getLosses() -> (generator: [Float], discriminator: [Float]) {
    return (generator: self.generatorLossHistory, discriminator: self.discriminatorLossHistory)
  }
  
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
  
  public func train(data: [TrainingData] = [],
                    epochCompleted: ((_ epoch: Int) -> ())? = nil,
                    complete: ((_ success: Bool) -> ())? = nil) {
    
    self.startTraining(data: data,
                       epochCompleted: epochCompleted,
                       complete: complete)
  }
  
  @discardableResult
  //we only expect ONE output neuron
  public func discriminate(_ input: [Float]) -> Float {
    guard let dis = self.discriminator else {
      return 0
    }
    
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

//MARK: Wasserstein Training
extension GAN {
  func wassersteinGenerator(fake: [TrainingData]) -> Float {
    let genOutput = self.disriminateOn(batch: fake, type: .real)
    return -1 * genOutput.loss
  }
  
  func wassersteinCritic(real: [TrainingData],
                         fake: [TrainingData],
                         withPenalty: Bool = false) -> Float {
    guard let dis = discriminator, real.count == fake.count else {
      return 0
    }
    
    var realLossAverage: Float = 0
    var fakeLossAverage: Float = 0
    var penaltyAverage: Float = 0
    
    for i in 0..<real.count {
      dis.zeroGradients()
      
      let realSample = real[i]
      let fakeSample = fake[i]
      let interSample = getInterpolated(real: realSample, fake: fakeSample)
      
      let realLoss = disriminateOn(batch: [realSample], type: .real).loss
      let fakeLoss = disriminateOn(batch: [fakeSample], type: .fake).loss
      
      if withPenalty {
        let interLoss = disriminateOn(batch: [interSample], type: .real).loss
        
        dis.backpropagate(with: [interLoss])
        
        if let networkGradients = dis.gradients()[safe: 1]?.flatMap({ $0 }) {
          let penalty = GradientPenalty.calculate(gradient: networkGradients)
          penaltyAverage += penalty / Float(real.count)
        }
      }

      realLossAverage += realLoss / Float(real.count)
      fakeLossAverage += fakeLoss / Float(fake.count)
    }
    
    let penalty = gradientPenaltyLambda * penaltyAverage
    gradientPenalty = penalty
    
    let criticLoss = fakeLossAverage - realLossAverage + penalty
    return criticLoss
  }
}

// MARK: Minimax Training
extension GAN {
  func minimaxGenerator(fake: [TrainingData]) -> Float {
    let genOutput = self.disriminateOn(batch: fake, type: .real)
    return -1 * genOutput.loss
  }
  
  func minimaxDiscriminator(real: [TrainingData],
                            fake: [TrainingData]) -> Float {
    
    //train discriminator on real data
    let realOutput = disriminateOn(batch: real, type: .real)

    //tran discriminator on new fake data
    let fakeOutput = disriminateOn(batch: fake, type: .fake)
    
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
}
