//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/21/21.
//

import Foundation
import Logger
import NumSwift

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
                                noise: [Float],
                                size: Int) -> [TrainingData] {
    var fakeData: [TrainingData] = []
    guard let gen = generator else {
      return []
    }
    
    for _ in 0..<size {
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
      //prepare data into batches
      for _ in 0..<self.criticTrainPerEpoch {
        //freeze the generator
        gen.trainable = false
        dis.trainable = true

        let noise = randomNoise()

        let realDataBatch = self.getRandomBatch(data: data)
        let fakeDataBatch = self.getGeneratedData(type: .fake, noise: noise)

        //zero out gradients before training discriminator on real image
        dis.zeroGradients()
        
        //train discriminator on real data
        let realOutput = self.disriminateOn(batch: realDataBatch, type: .real)

        //zero out gradients before training discriminator on fake image
        dis.zeroGradients()

        //tran discriminator on new fake data
        let fakeOutput = self.disriminateOn(batch: fakeDataBatch, type: .fake)
        
        if self.lossFunction == .minimax {
          let realLoss = realOutput.loss
          let fakeLoss = fakeOutput.loss
          
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
          
          let averageRealOut = realOutput.loss
          let averageFakeOut = fakeOutput.loss
          
          let lambda: Float = gradientPenaltyLambda
          let penalty = self.gradientPenalty(realData: realDataBatch,
                                             fakeData: fakeDataBatch)

          self.gradientPenalty = penalty * lambda
          
          //negative because Neuron only minimizes gradients so we want to revert the sign so W - lr * -g becomes W + lr * g
          self.discriminatorLoss = (averageFakeOut - averageRealOut + gradientPenalty)
          
          //backprop discrimator
          dis.backpropagate(with: [discriminatorLoss])
        }
        
        //adjust weights AFTER calculating gradients
        //figure out batch size
        dis.adjustWeights(batchSize: 1)
      }
      
      for _ in 0..<self.generatorTrainPerEpoch {
        //freeze the generator
        gen.trainable = true
        dis.trainable = false
        
        let noise = randomNoise()
        //zero out gradients before training generator
        gen.zeroGradients()
        dis.zeroGradients()

        //train generator on newly trained discriminator
        let generatedData = self.getGeneratedData(type: .real, noise: noise, size: self.batchSize)
        let genOutput = self.disriminateOn(batch: generatedData, type: .real)
        
        if self.lossFunction == .minimax {
          let averageLoss = genOutput.loss
          
          //we want to maximize lossfunction log(D(G(z))
          //negative because the Neuron only supports MINIMIZING gradients
          let genLoss = -1 * averageLoss
          
          self.generatorLoss = genLoss
          
        } else if self.lossFunction == .wasserstein {
          let averageGenLoss = genOutput.loss
          
          //minimize gradients
          self.generatorLoss = -averageGenLoss
        }
        
        //backprop discrimator
        let firstLayerDeltas = dis.backpropagate(with: [self.generatorLoss]).firstLayerDeltas
                
        //get discriminator gradients for each generator parameter first
        gen.backpropagate(with: firstLayerDeltas)
        gen.adjustWeights(batchSize: 1) //figure out batch size?
      }
      
      epochCompleted?(i)
      
      if self.checkGeneratorValidation(for: i) {
        return
      }
    }
    
    self.log(type: .message, priority: .alwaysShow, message: "GAN Training complete")

    complete?(false)
  }
  
  
//  private func gradentPenalty(real: TrainingData, fake: TrainingData) -> Float {
//    guard let dis = self.discriminator else {
//      return 0
//    }
//
//    defer {
//      dis.zeroGradients()
//    }
//
//    var gradients: [Float] = []
//
//    let epsilon = Float.random(in: 0...1)
//
//    var inter: [Float] = []
//    let realNew = real.data
//    let fakeNew = fake.data
//
//    guard realNew.count == fakeNew.count else {
//      return 0
//    }
//
//    inter = (realNew * epsilon) + (fakeNew * (1 - epsilon))
//
//    let output = self.discriminate(inter)
//    let loss = self.lossFunction.loss(.real, value: output)
//
//    dis.backpropagate(with: [loss])
//
//    //skip first layer gradients
//    if let networkGradients = dis.gradients()[safe: 1]?.flatMap({ $0 }) {
//      gradients = networkGradients
//    }
//
//    let gradientNorm = gradients.sumOfSquares
//    let center = self.gradientPenaltyCenter
//
//    let penalty = gradientNorm.map { pow((sqrt($0) - center), 2) }.sum / (Float(gradientNorm.count) + 1e-8)
//
//  }
  
  private func gradientPenalty(realData: [TrainingData], fakeData: [TrainingData]) -> Float {
    guard let dis = self.discriminator else {
      return 0
    }
    
    defer {
      dis.zeroGradients()
    }
        
    var gradients: [[Float]] = []
    
    let real = realData
    let fake = fakeData
    
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
        
        inter = (realNew * epsilon) + (fakeNew * (1 - epsilon))
      }
      
      let output = self.discriminate(inter)
      let loss = self.lossFunction.loss(.real, value: output)

      dis.backpropagate(with: [loss])
      
      //skip first layer gradients
      if let networkGradients = dis.gradients()[safe: 1]?.flatMap({ $0 }) {
        gradients.append(networkGradients)
      }
    }
    
    let gradientNorm = gradients.map { $0.sumOfSquares }

    let center = self.gradientPenaltyCenter
    
    let penalty = gradientNorm.map { pow((sqrt($0) - center), 2) }.sum / (Float(gradientNorm.count) + 1e-8)
    return penalty
  }
    
  private func disriminateOn(batch: [TrainingData], type: GANTrainingType) -> (loss: Float, output: [Float]) {
    //train on each sample
    var outputs: [Float] = []
    
    var averageLoss: Float = 0
    
    for i in 0..<batch.count {
      //get sample from generator
      let sample = batch[i].data

      //feed sample
      let output = self.discriminate(sample)
      
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
                    singleStep: Bool = false,
                    epochCompleted: ((_ epoch: Int) -> ())? = nil,
                    complete: ((_ success: Bool) -> ())? = nil) {
    
    self.startTraining(data: data,
                       singleStep: singleStep,
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
