//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/21/21.
//

import Foundation
import Logger

public enum GANType {
  case generator, discriminator
}

public enum GANTrainingType: String {
  case real, fake, generator

}

public enum GANLossFunction {
  case minimax
  //-1 for real : 1 for fake in wasserstein
  
  public func label(type: GANTrainingType) -> Float {
    switch self {
    case .minimax:
      switch type {
      case .real, .generator:
        return 1.0
      case .fake:
        return 0
      }
    }
  }
  
  public func loss(_ type: GANTrainingType, value: Float) -> Float {
    switch self {
    case .minimax:
      switch type {
      case .fake:
        return log(1 - value)
      case .generator:
        return log(value)
      case .real:
        return log(value)
      }
    }
  }
}

public class GAN: Logger {
  private var generator: Brain?
  private var discriminator: Brain?
  private var batchSize: Int
  private var criticTrainPerEpoch: Int = 5

  public var epochs: Int
  public var logLevel: LogLevel = .none
  public var randomNoise: () -> [Float]
  public var validateGenerator: (_ output: [Float]) -> Bool
  public var discriminatorNoiseFactor: Float = 0.1
  public var lossFunction: GANLossFunction = .minimax
  public var discriminatorLoss: Float = 0
  public var generatorLoss: Float = 0
  public var weightConstraints: ClosedRange<Float>? = nil

  //MARK: Init
  public init(generator: Brain? = nil,
              discriminator: Brain? = nil,
              epochs: Int,
              criticTrainPerEpoch: Int = 5,
              batchSize: Int) {
    
    self.epochs = epochs
    self.batchSize = batchSize
    self.criticTrainPerEpoch = criticTrainPerEpoch
    self.generator = generator
    self.discriminator = discriminator
    
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

  //return data batched into size
  private func getBatchedRandomData(data: [TrainingData]) -> [[TrainingData]] {
    let random = data.shuffled()
    let preBatched = random.batched(into: self.batchSize)
    return preBatched
  }
  
  private func checkGeneratorValidation(for epoch: Int) -> Bool {
    if epoch % 5 == 0 {
      return self.validateGenerator(getGeneratedSample())
    }
    
    return false
  }

  //fake data that is FAKE not acting as REAL like when training the generator
  private func getFakeData(_ count: Int) -> [TrainingData] {
    var fakeData: [TrainingData] = []
    for _ in 0..<count {
      let sample = self.getGeneratedSample()
      
      let label = lossFunction.label(type: .fake)
      
      var training = TrainingData(data: sample, correct: [label])
      if self.discriminatorNoiseFactor < 1.0 {
        let factor = min(1.0, max(0.0, self.discriminatorNoiseFactor))
        training = TrainingData(data: sample, correct: [Float.random(in: (label - factor)...label)])
      }
      fakeData.append(training)
    }
    
    return fakeData
  }

  private func startTraining(data: [TrainingData],
                             singleStep: Bool = false,
                             complete: ((_ complete: Bool) -> ())? = nil) {
    
    guard let dis = self.discriminator, let gen = self.generator else {
      return
    }

    guard data.count > 0 else {
      return
    }

    let epochs = singleStep ? 1 : self.epochs
    
    //control epochs locally
    dis.epochs = 1
    
    //prepare data into batches
    let realData = self.getBatchedRandomData(data: data)
    
    self.log(type: .message, priority: .alwaysShow, message: "Training started...")
    
    for i in 0..<epochs {
      
      for _ in 0..<self.criticTrainPerEpoch {
        //get next batch of real data
        let realDataBatch = realData.randomElement() ?? []
        
        //train discriminator on real data combined with fake data
        let realLoss = self.trainDiscriminator(data: realDataBatch, type: .real) / Float(self.batchSize)
        
        //get next batch of fake data by generating new fake data
        let fakeDataBatch = self.getFakeData(self.batchSize)
        
        //tran discriminator on new fake data generated after epoch
        let fakeLoss = self.trainDiscriminator(data: fakeDataBatch, type: .fake) / Float(self.batchSize)
        
        let averageTotalLoss = (realLoss + fakeLoss)
        
        //figure out how to make this more modular than hard coding addition for minimax
        self.discriminatorLoss = averageTotalLoss
        
        //backprop discrimator
        dis.backpropagate(with: [discriminatorLoss])
        
        //adjust weights AFTER calculating gradients
        dis.adjustWeights(self.weightConstraints)
      }
      
      //train generator on newly trained discriminator
      let genLoss = (self.trainGenerator() / Float(self.batchSize))
      
      self.generatorLoss = genLoss
      
      //backprop discrimator
      dis.backpropagate(with: [self.generatorLoss])
      
      //get deltas from discrimator
      if dis.lobes.count > 1 {
        let deltas = dis.lobes[1].deltas()
        
        gen.backpropagate(with: deltas)
        
        //adjust weights of generator
        gen.adjustWeights()
      }
      
      if self.checkGeneratorValidation(for: i) {
        return
      }
      self.discriminatorLoss = 0
      self.generatorLoss = 0
    }
    
    self.log(type: .message, priority: .alwaysShow, message: "GAN Training complete")

    complete?(false)
  }
  
  //single step operation only
  private func trainGenerator() -> Float {
    var averageLoss: Float = 0
    //train on each sample
    //calculate on each batch then back prop
    for _ in 0..<self.batchSize {
      //get sample from generator
      let sample = self.getGeneratedSample()
     // let trainingData = TrainingData(data: sample, correct: [label])

      //feed sample
      let output = self.discriminate(sample)
        
      let first = output.first ?? 0
      
      let loss = self.lossFunction.loss(.generator, value: first)
      
      averageLoss += loss
    }
    
    let loss = averageLoss
    
    return loss
  }
    
  private func trainDiscriminator(data: [TrainingData], type: GANTrainingType) -> Float {
    //train on each sample
    
    var averageLoss: Float = 0
    
    for i in 0..<data.count {
      //get sample from generator
      let sample = data[i].data

      //feed sample
      let output = self.discriminate(sample)
      
      let first = output.first ?? 0
      
      //get loss for type
      let loss = self.lossFunction.loss(type, value: first)
      
      //add losses together
      averageLoss += loss
    }
    
    //get average loss over batch
    return averageLoss
  }
  
//MARK: Public Functions
  
  public func add(generator gen: Brain) {
    self.generator = gen
    gen.compile()
  }
  
  public func add(discriminator dis: Brain) {
    self.discriminator = dis
    dis.compile()
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
    guard let dis = self.discriminator else {
      return []
    }
    
    let output = dis.feed(input: input)
    return output
  }
  
  public func getGeneratedSample() -> [Float] {
    guard let gen = self.generator else {
      return []
    }
    
    return gen.feed(input: randomNoise())
  }
  
}
