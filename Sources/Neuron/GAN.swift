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
  private var discriminatorLossHistory: [Float] = []
  private var generatorLossHistory: [Float] = []

  public var epochs: Int
  public var logLevel: LogLevel = .none
  public var randomNoise: () -> [Float]
  public var validateGenerator: (_ output: [Float]) -> Bool
  public var discriminatorNoiseFactor: Float?
  public var weightConstraints: ClosedRange<Float>? = nil
  public var lossFunction: GANLossFunction = .minimax
  public var discriminatorLoss: Float = 0 {
    didSet {
      self.discriminatorLossHistory.append(discriminatorLoss)
    }
  }
  public var generatorLoss: Float = 0 {
    didSet {
      self.generatorLossHistory.append(generatorLoss)
    }
  }

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
    let filtered = preBatched.filter({ $0.count == self.batchSize })
    return filtered
  }
  
  private func checkGeneratorValidation(for epoch: Int) -> Bool {
    if epoch % 5 == 0 {
      return self.validateGenerator(getGeneratedSample())
    }
    
    return false
  }

  private func getGeneratedData(_ count: Int, type: GANTrainingType) -> [TrainingData] {
    var fakeData: [TrainingData] = []
    for _ in 0..<count {
      let sample = self.getGeneratedSample()
      
      let label = lossFunction.label(type: type)
      
      var training = TrainingData(data: sample, correct: [label])
      //assuming the label is 1.0 or greater
      //we need to reverse if label is <= 0
      if let noise = self.discriminatorNoiseFactor, noise > 0, noise < 1 {
        //cap factor between 0 and 1
        let factor = min(1.0, max(0.0, noise))
        let min = min(label, abs(label - factor))
        let max = max(label, abs(label - factor))
        
        training = TrainingData(data: sample, correct: [Float.random(in: (min...max))])
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

    
    self.log(type: .message, priority: .alwaysShow, message: "Training started...")
    
    for i in 0..<epochs {
      
      //prepare data into batches
      let realData = self.getBatchedRandomData(data: data)
      
      for _ in 0..<self.criticTrainPerEpoch {
        //get next batch of real data
        let realDataBatch = realData.randomElement() ?? []
        
        //train discriminator on real data combined with fake data
        let realLoss = self.trainOn(data: realDataBatch, type: .real).reduce(0, +)
        
        //get next batch of fake data by generating new fake data
        let fakeDataBatch = self.getGeneratedData(self.batchSize, type: .fake)
        
        //tran discriminator on new fake data generated after epoch
        let fakeLoss = self.trainOn(data: fakeDataBatch, type: .fake).reduce(0, +)
        
        //adding real and fake based on minimax loss function of log(D(x)) + log(D(G(z)))
        let totalSumLoss = realLoss + fakeLoss
        //we are taking the sum of all instances of the minibatch and dividing by batch size
        //to get average loss
        //negative because the Neuron only supports MINIMIZING gradients
        let averageTotalLoss = -1 * (totalSumLoss / Float(2 * batchSize))
        
        //figure out how to make this more modular than hard coding addition for minimax
        self.discriminatorLoss = averageTotalLoss
        
        //backprop discrimator
        dis.backpropagate(with: [discriminatorLoss])
        
        //adjust weights AFTER calculating gradients
        dis.adjustWeights(self.weightConstraints)
      }
      
      //train generator on newly trained discriminator
      let realFakeFata = self.getGeneratedData(self.batchSize, type: .real)
      let sumOfGenLoss = self.trainOn(data: realFakeFata, type: .generator).reduce(0, +)
      
      //we want to maximize lossfunction log(D(G(z))
      //negative because the Neuron only supports MINIMIZING gradients
      let genLoss = -1 * (sumOfGenLoss / Float(self.batchSize))
      
      self.generatorLoss = genLoss
      
      //backprop discrimator
      dis.backpropagate(with: [self.generatorLoss])
      
      //get deltas from discrimator
      if let deltas = dis.lobes.first(where: { $0.deltas().count > 0 })?.deltas() {
        gen.backpropagate(with: deltas)
        
        //adjust weights of generator
        gen.adjustWeights()
      }
      
      if self.checkGeneratorValidation(for: i) {
        return
      }
    }
    
    self.log(type: .message, priority: .alwaysShow, message: "GAN Training complete")

    complete?(false)
  }
    
  private func trainOn(data: [TrainingData], type: GANTrainingType) -> [Float] {
    //train on each sample
    
    var losses: [Float] = []
    
    for i in 0..<data.count {
      //get sample from generator
      let sample = data[i].data

      //feed sample
      let output = self.discriminate(sample)
      
      let first = output.first ?? 0
      
      //get loss for type
      let loss = self.lossFunction.loss(type, value: first)
      
      //add losses together
      losses.append(loss)
    }
    
    //get average loss over batch
    return losses
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
