//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/23/22.
//

import Foundation
import NumSwift

public class GAN<N: TensorNumeric> {
  public enum TrainingType: String {
    case real, fake
  }

  public var generator: BaseOptimizer<N>
  public var discriminator: BaseOptimizer<N>
  public var noise: () -> Tensor<N>
  public var onEpochCompleted: ((_ epoch: Int) -> ())? = nil
  public var validateGenerator: ((_ output: Tensor<N>) -> ())? = nil
  public var onCompleted: (() -> ())? = nil
  public private(set) var lossFunction: LossFunction = .minimaxBinaryCrossEntropy
  public private(set) var fakeLabel: Tensor<N>.Scalar = 0
  public private(set) var realLabel: Tensor<N>.Scalar = 1
  public var epochs: Int
  
  internal let batchSize: Int
  let threadWorkers: Int
  private let discriminatorSteps: Int
  private let generatorSteps: Int
  private let discriminatorNoiseFactor: Tensor<N>.Scalar?
  private let validationFrequency: Int
  
  public init(generator: BaseOptimizer<N>,
              discriminator: BaseOptimizer<N>,
              epochs: Int = 100,
              batchSize: Int,
              discriminatorSteps: Int = 1,
              generatorSteps: Int = 1,
              discriminatorNoiseFactor: Tensor<N>.Scalar? = nil,
              threadWorkers: Int = 16,
              validationFrequency: Int = 5) {
    self.generator = generator
    self.discriminator = discriminator
    self.epochs = epochs
    self.batchSize = batchSize
    self.threadWorkers = threadWorkers
    self.discriminatorSteps = discriminatorSteps
    self.generatorSteps = generatorSteps
    self.discriminatorNoiseFactor = discriminatorNoiseFactor
    self.validationFrequency = validationFrequency
    
    self.generator.workers = threadWorkers
    self.discriminator.workers = threadWorkers
    
    self.noise = {
      var noise: [Tensor<N>.Scalar] = []
      for _ in 0..<10 {
        noise.append(Tensor<N>.Scalar.random(in: 0...1))
      }
      return Tensor<N>(noise)
    }
    
  }
  
  public func fit(_ data: [DatasetModel], _ validation: [DatasetModel]) {
    let batches = data.batched(into: batchSize)
    let _ = validation.batched(into: batchSize)
    
    let valBatches = validation.batched(into: batchSize)

    //epochs
    var trainingBatches: [(data: [Tensor<N>], labels: [Tensor<N>])] = []
    batches.forEach { b in
      if b.count == batchSize {
        let trainingSet = splitDataset(b)
        trainingBatches.append(trainingSet)
      }
    }

    var validationBatches: [(data: [Tensor<N>], labels: [Tensor<N>])] = []
    valBatches.forEach { b in
      if b.count == batchSize {
        let valSet = splitDataset(validation)
        validationBatches.append(valSet)
      }
    }
    
    for e in 0..<epochs {
      var i = 0
      var dataIterations = 0
      while dataIterations < trainingBatches.count {
        
        if i % validationFrequency == 0 {
          DispatchQueue.main.async {
            self.validateGenerator?(self.generate())
          }
          
          i = 1
        }
        
        //train discriminator
        for _ in 0..<discriminatorSteps {
          if dataIterations < trainingBatches.count {
            let b = trainingBatches[dataIterations]
            generator.isTraining = false
            discriminator.isTraining = true
            
            discriminatorStep(b.data, labels: b.labels)
            discriminator.metricsReporter?.report()
          }
          dataIterations += 1
        }
        
        //train generator
        for _ in 0..<generatorSteps {
          generator.isTraining = true
          discriminator.isTraining = false
          
          generatorStep()
          generator.metricsReporter?.report()
        }
        
        i += 1
      }
      onEpochCompleted?(e)
    }
    
    onCompleted?()
  }
  
  public func generate() -> Tensor<N> {
    self.generator.isTraining = false
    let out = generator([noise()])
    return out.first?.detached() ?? Tensor<N>()
  }
  
  public func discriminate(_ input: [Tensor<N>]) -> Tensor<N> {
    let out = discriminator(input)
    return out.first ?? Tensor<N>()
  }
  
  @discardableResult
  public func export(overrite: Bool = false, compress: Bool = false) -> (discriminator: URL?, generator: URL?) {
    var urls: (URL?, URL?) = (nil, nil)
    if let generatorS = generator.trainable as? Sequential,
       let discS = discriminator.trainable as? Sequential {
      
      let additional = overrite == false ? "-\(Date().timeIntervalSince1970)" : ""
      
      let dUrl = ExportHelper.getModel(filename: "discriminator\(additional)", compress: compress, model: discS)
      let gUrl = ExportHelper.getModel(filename: "generator\(additional)", compress: compress, model: generatorS)
      
      urls = (dUrl, gUrl)
    }
    
    return urls
  }
  
  internal func discriminatorStep(_ real: [Tensor<N>], labels: [Tensor<N>]) {
    discriminator.zeroGradients()
    
    let realOutput = trainOn(real, labels: labels)
    
    discriminator.apply(realOutput.gradients)
    
    let fake = getGenerated(.fake, detatch: true, count: batchSize)
    let fakeOutput = trainOn(fake.data, labels: fake.labels)
    
    discriminator.apply(fakeOutput.gradients)
    
    let realLoss = realOutput.loss
    let fakeLoss = fakeOutput.loss
    
    //adding real and fake based on minimax loss function of log(D(x)) + log(1 - D(G(z)))
    let totalSumLoss = (realLoss + fakeLoss)
          
    discriminator.metricsReporter?.update(metric: .criticLoss, value: totalSumLoss)
    discriminator.metricsReporter?.update(metric: .realImageLoss, value: realLoss)
    discriminator.metricsReporter?.update(metric: .fakeImageLoss, value: fakeLoss)
    
    discriminator.step()
  }
  
  internal func generatorStep() {
    generator.zeroGradients()
    
    let generatedData = getGenerated(.real, count: batchSize)
    let output = trainOn(generatedData.data, labels: generatedData.labels)
    
    let loss = output.loss
    generator.metricsReporter?.update(metric: .generatorLoss, value: loss)
    
    generator.apply(output.gradients)
    generator.step()
  }
  
  internal func trainOn(_ batch: [Tensor<N>],
                        labels: [Tensor<N>],
                        requiresGradients: Bool = true) -> Optimizer.Output {
    discriminator.fit(batch,
                      labels: labels,
                      lossFunction: lossFunction,
                      validation: false,
                      requiresGradients: requiresGradients)
  }
  
  internal func getGenerated(_ label: TrainingType, detatch: Bool = false, count: Int) -> (data: [Tensor<N>], labels: [Tensor<N>]) {
    var fakeData: [Tensor<N>] = [Tensor<N>](repeating: Tensor<N>(), count: count)
    var fakeLabels: [Tensor<N>] = [Tensor<N>](repeating: Tensor<N>(), count: count)

    Array(0..<count).concurrentForEach(workers: min(Constants.maxWorkers, Int(ceil(Double(count) / Double(4))))) { _, index in
      var sample = self.generator([self.noise()])[safe: 0, Tensor<N>()]
      
      if detatch {
        sample = sample.detached()
      }
      
      let localLabelValue = label == .fake ? self.fakeLabel : self.realLabel
      var dataLabel = Tensor<N>([localLabelValue])
      var training = sample
      //assuming the label is 1.0 or greater
      //we need to reverse if label is <= 0
      if let noise = self.discriminatorNoiseFactor, noise > 0, noise < 1 {
        //cap factor between 0 and 1
        let factor = min(1.0, max(0.0, noise))
        let min = min(localLabelValue, abs(localLabelValue - factor))
        let max = max(localLabelValue, abs(localLabelValue - factor))
        
        training = sample
        dataLabel = Tensor<N>([Tensor<N>.Scalar.random(in: (min...max))])
      }
      
      fakeData[index] = training
      fakeLabels[index] = dataLabel
    }

    return (fakeData, fakeLabels)
  }

  private func splitDataset(_ data: [DatasetModel]) -> (data: [Tensor<N>], labels: [Tensor<N>]) {
    var labels: [Tensor<N>] = []
    var input: [Tensor<N>] = []
    
    data.forEach { d in
      labels.append(d.label)
      input.append(d.data)
    }
    
    return (input, labels)
  }
}
