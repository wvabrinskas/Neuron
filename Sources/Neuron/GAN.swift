//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/21/21.
//

import Foundation
import Logger

public class GAN: Logger {
  private var generator: Brain?
  private var discriminator: Brain?
  private var batchSize: Int
  private var lossTreshold: Float
  
  public var epochs: Int
  public var logLevel: LogLevel = .none
  public var randomNoise: () -> [Float]
  public var validateGenerator: (_ output: [Float]) -> Bool

  //MARK: Init
  public init(generator: Brain? = nil,
              discriminator: Brain? = nil,
              epochs: Int,
              lossThreshold: Float = 0.001,
              batchSize: Int) {
    
    self.epochs = epochs
    self.batchSize = batchSize
    self.lossTreshold = lossThreshold
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

  private func getFakeData(_ count: Int) -> [TrainingData] {
    var fakeData: [TrainingData] = []
    for _ in 0..<count {
      let sample = self.getGeneratedSample()
      let training = TrainingData(data: sample, correct: [0.0])
      fakeData.append(training)
    }
    
    return fakeData
  }
  
  private func startTraining(data: [TrainingData],
                             singleStep: Bool = false,
                             complete: ((_ complete: Bool) -> ())? = nil) {
    
    guard let dis = self.discriminator else {
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
      if self.checkGeneratorValidation(for: i) {
        return
      }
      
      //get next batch of real data
      let realDataBatch = realData.randomElement() ?? []
      
      //train discriminator on real data combined with fake data
      dis.train(data: realDataBatch)
      
      //get next batch of fake data by generating new fake data
      let fakeDataBatch = self.getFakeData(self.batchSize)
      
      //tran discriminator on new fake data generated after epoch
      dis.train(data: fakeDataBatch)
      
      //train generator on newly trained discriminator
      self.trainGenerator()
    }
    
    self.log(type: .message, priority: .alwaysShow, message: "GAN Training complete")

    complete?(false)
  }
  
  //single step operation only
  private func trainGenerator() {
    guard let dis = self.discriminator, let gen = self.generator else {
      return
    }
    
    //train on each sample
    for _ in 0..<self.batchSize {
      //get sample from generator
      let sample = self.getGeneratedSample()
      let trainingData = TrainingData(data: sample, correct: [1.0])

      //feed sample
      let output = self.discriminate(sample)
      
      //calculate loss at discrimator
      let loss = dis.calcAverageLoss(output, correct: [1.0])
      gen.loss.append(loss)
      
      
      self.log(type: .message, priority: .low, message: "Generator loss: \(loss)")
      
      //calculate loss at last layer for discrimator
      //we want it to be real so correct is [1.0, 0.0] [real, fake]
      dis.setOutputDeltas(trainingData.correct)

      //backprop discrimator
      dis.backpropagate()
      
      //get deltas from discrimator
      if dis.lobes.count > 1 {
        let deltas = dis.lobes[1].deltas()
        gen.backpropagate(with: deltas)
        
        //adjust weights of generator
        gen.adjustWeights()
      }
    }
    //repeat
    
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
