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
  case real, fake

}

public enum GANLossFunction {
  case minimax
  case wasserstein
  
  public func label(type: GANTrainingType) -> Float {
    switch self {
    case .minimax:
      switch type {
      case .real:
        return 1.0
      case .fake:
        return 0
      }
      
    case .wasserstein:
      switch type {
      case .real:
        return 1
      case .fake:
        return -1
      }
    }
  }
  
  public func loss(_ type: GANTrainingType, value: Float) -> Float {
    switch self {
    case .minimax:
      switch type {
      case .fake:
        return log(1 - value)
      case .real:
        return log(value)
      }
    case .wasserstein:
      return self.label(type: type) * value
    }
  }
}

public class GAN: Logger {
  private var generator: Brain?
  private var discriminator: Brain?
  private var batchSize: Int
  private var criticTrainPerEpoch: Int = 5
  private var generatorTrainPerEpoch: Int = 5
  private var discriminatorLossHistory: [Float] = []
  private var generatorLossHistory: [Float] = []
  private var gradientPenaltyHistory: [Float] = []
  private var gradientPenaltyCenter: Float = 0
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
              generatorTrainPerEpoch: Int = 5,
              gradientPenaltyCenter: Float = 0,
              gradientPenaltyLambda: Float = 10,
              batchSize: Int) {
    
    self.epochs = epochs
    self.batchSize = batchSize
    self.criticTrainPerEpoch = criticTrainPerEpoch
    self.generatorTrainPerEpoch = generatorTrainPerEpoch
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

  private func getRandomBatch(data: [TrainingData]) -> [TrainingData] {
    var newData: [TrainingData] = []
    for _ in 0..<self.batchSize {
      if let element = data.randomElement() {
        newData.append(element)
      }
    }
    return newData
  }

  private func getGeneratedData(type: GANTrainingType,
                                noise: [Float]) -> [TrainingData] {
    var fakeData: [TrainingData] = []
    guard let gen = generator else {
      return []
    }
    
    for _ in 0..<self.batchSize {
      let sample = gen.feed(input: noise)
      
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
                             epochCompleted: ((_ epoch: Int) -> ())? = nil,
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
      
      let noise = randomNoise()
      
      //prepare data into batches
      let realData = self.getRandomBatch(data: data)
      
      for _ in 0..<self.criticTrainPerEpoch {
        //zero out gradients before training discriminator
        dis.zeroGradients()
        
        //get next batch of real data
        let realDataBatch = self.getRandomBatch(data: realData)
        //train discriminator on real data combined with fake data
        let realOutput = self.trainOn(data: realDataBatch, type: .real)
        
        //get next batch of fake data by generating new fake data
        let fakeDataBatch = self.getGeneratedData(type: .fake, noise: noise)

        //tran discriminator on new fake data generated after epoch
        let fakeOutput = self.trainOn(data: fakeDataBatch, type: .fake)
        
        if self.lossFunction == .minimax {
          let realLoss = realOutput.loss.sum / Float(self.batchSize)
          let fakeLoss = fakeOutput.loss.sum / Float(self.batchSize)
          
          //backprop discrimator
          dis.backpropagate(with: [realLoss])
          dis.backpropagate(with: [fakeLoss])

          //adding real and fake based on minimax loss function of log(D(x)) + log(D(G(z)))
          let totalSumLoss = realLoss + fakeLoss
          //we are taking the sum of all instances of the minibatch and dividing by batch size
          //to get average loss
          //negative because the Neuron only supports MINIMIZING gradients
          let averageTotalLoss = -1 * totalSumLoss
          
          //figure out how to make this more modular than hard coding addition for minimax
          self.discriminatorLoss = averageTotalLoss
          
        } else if self.lossFunction == .wasserstein {
          
          let averageRealOut = (realOutput.loss.sum / Float(self.batchSize))
          let averageFakeOut = (fakeOutput.loss.sum / Float(self.batchSize))
          
          let lambda: Float = gradientPenaltyLambda
          let penalty = lambda * self.gradientPenalty(realData: realData)
          
          self.gradientPenalty = penalty
          
          self.discriminatorLoss = (averageRealOut - averageFakeOut) + penalty
          
          //backprop discrimator
          dis.backpropagate(with: [discriminatorLoss])
        }
        
        //adjust weights AFTER calculating gradients
        dis.adjustWeights()
      }
      
      for _ in 0..<self.generatorTrainPerEpoch {
        
        //zero out gradients before training generator
        gen.zeroGradients()
        dis.zeroGradients()

        //train generator on newly trained discriminator
        let realFakeData = self.getGeneratedData(type: .fake, noise: noise)
        let genOutput = self.trainOn(data: realFakeData, type: .fake)
        
        if self.lossFunction == .minimax {
          let sumOfGenLoss = genOutput.loss.sum
          
          //we want to maximize lossfunction log(D(G(z))
          //negative because the Neuron only supports MINIMIZING gradients
          let genLoss = -1 * (sumOfGenLoss / Float(self.batchSize))
          
          self.generatorLoss = genLoss
          
        } else if self.lossFunction == .wasserstein {
          let sumOfGenLoss = genOutput.loss.sum
          let averageGenLoss = sumOfGenLoss / Float(self.batchSize)
          
          //minimize gradients
          self.generatorLoss = -1 * averageGenLoss
        }
        
        //backprop discrimator
        dis.backpropagate(with: [self.generatorLoss])
                
        //get discriminator gradients for each generator parameter first
        if let firstLayerGradients = dis.lobes.first(where: { $0.deltas().count > 0 })?.deltas() {
          gen.backpropagate(with: firstLayerGradients)
          gen.adjustWeights()
        }
  
      }
      
      epochCompleted?(i)
      
      if self.checkGeneratorValidation(for: i) {
        return
      }
    }
    
    self.log(type: .message, priority: .alwaysShow, message: "GAN Training complete")

    complete?(false)
  }
  
  private func gradientPenalty(realData: [TrainingData]) -> Float {
    guard let dis = self.discriminator else {
      return 0
    }
    
    let noise = self.randomNoise()
    
    var gradients: [[Float]] = []
    
    let real = self.getRandomBatch(data: realData)
    let fake = self.getGeneratedData(type: .real, noise: noise)
    
    for i in 0..<self.batchSize {
      dis.zeroGradients()
      
      let epsilon = Float.random(in: 0...1)
      var inter: [Float] = []
      if i < real.count && i < fake.count {
        let realNew = real[i].data
        let fakeNew = fake[i].data
        
        guard realNew.count == fakeNew.count else {
          return 0
        }
        
        for i in 0..<realNew.count {
          let realVal = realNew[i] * epsilon
          let fakeVal = fakeNew[i] * (1 - epsilon)
          inter.append(realVal + fakeVal)
        }
      }
      
      let output = self.discriminate(inter).first ?? 0
      let loss = self.lossFunction.label(type: .real) - output

      dis.backpropagate(with: [loss])
      
      //skip first layer gradients
      if let firstLayerGradients = dis.lobes.first(where: { $0.deltas().count > 0 })?.deltas() {
        gradients.append(firstLayerGradients)
      }
    }
    
    let squared = gradients.map { $0.sumOfSquares }

    let center = self.gradientPenaltyCenter
    
    let penalty = squared.map { pow((sqrt($0) - center), 2) }.sum / (Float(squared.count) + 1e-8)
    return penalty
  }
    
  private func trainOn(data: [TrainingData], type: GANTrainingType) -> (loss: [Float], output: [Float]) {
    //train on each sample
    
    var losses: [Float] = []
    var outputs: [Float] = []
    
    for i in 0..<data.count {
      //get sample from generator
      let sample = data[i].data

      //feed sample
      let output = self.discriminate(sample)
      
      let first = output.first ?? 0
  
      //append outputs for wasserstein
      outputs.append(first)
      
      //get loss for type
      let loss = self.lossFunction.loss(type, value: first)
      
      //add losses together
      losses.append(loss)
    }
    
    //get average loss over batch
    return (loss: losses, output: outputs)
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
                    epochCompleted: ((_ epoch: Int) -> ())? = nil,
                    complete: ((_ success: Bool) -> ())? = nil) {
    
    self.startTraining(data: data,
                       singleStep: singleStep,
                       epochCompleted: epochCompleted,
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
