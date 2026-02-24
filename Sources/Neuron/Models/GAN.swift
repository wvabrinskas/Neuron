//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/23/22.
//

import Foundation
import NumSwift

open class GAN {
  /// Indicates whether training data is real or generated (fake).
  public enum TrainingType: String {
    case real, fake
  }

  /// The optimizer responsible for updating the generator network.
  public var generator: Optimizer
  /// The optimizer responsible for updating the discriminator network.
  public var discriminator: Optimizer
  /// A closure that produces a noise tensor used as input to the generator.
  public var noise: () -> Tensor
  /// An optional closure called at the end of each training epoch.
  /// - Parameter epoch: The index of the completed epoch.
  public var onEpochCompleted: ((_ epoch: Int) -> ())? = nil
  /// An optional closure called periodically to validate generator output.
  /// - Parameter output: The tensor produced by the generator for validation.
  public var validateGenerator: ((_ output: Tensor) -> ())? = nil
  /// An optional closure called when training has fully completed.
  public var onCompleted: (() -> ())? = nil
  public private(set) var lossFunction: LossFunction = .minimaxBinaryCrossEntropy
  public private(set) var fakeLabel: Tensor.Scalar = 0
  public private(set) var realLabel: Tensor.Scalar = 1
  /// The number of epochs for which the GAN will be trained.
  public var epochs: Int
  
  internal let batchSize: Int
  private let discriminatorSteps: Int
  private let generatorSteps: Int
  private let discriminatorNoiseFactor: Tensor.Scalar?
  private let validationFrequency: Int
  internal let threadWorkers = Constants.maxWorkers
  
  /// Creates a GAN trainer with generator/discriminator optimizers.
  ///
  /// - Parameters:
  ///   - generator: Optimizer controlling generator updates.
  ///   - discriminator: Optimizer controlling discriminator updates.
  ///   - epochs: Number of training epochs.
  ///   - batchSize: Batch size used for both networks.
  ///   - discriminatorSteps: Number of discriminator updates per cycle.
  ///   - generatorSteps: Number of generator updates per cycle.
  ///   - discriminatorNoiseFactor: Optional label-noise factor for discriminator labels.
  ///   - validationFrequency: Interval for calling `validateGenerator`.
  public init(generator: Optimizer,
              discriminator: Optimizer,
              epochs: Int = 100,
              batchSize: Int,
              discriminatorSteps: Int = 1,
              generatorSteps: Int = 1,
              discriminatorNoiseFactor: Tensor.Scalar? = nil,
              validationFrequency: Int = 5) {
    self.generator = generator
    self.discriminator = discriminator
    self.epochs = epochs
    self.batchSize = batchSize
    self.discriminatorSteps = discriminatorSteps
    self.generatorSteps = generatorSteps
    self.discriminatorNoiseFactor = discriminatorNoiseFactor
    self.validationFrequency = validationFrequency
    
    self.noise = {
      var noise: [Tensor.Scalar] = []
      for _ in 0..<10 {
        noise.append(Tensor.Scalar.random(in: 0...1))
      }
      return Tensor(noise)
    }
    
  }
  
  /// Trains GAN components on the provided dataset.
  ///
  /// - Parameters:
  ///   - data: Real training samples.
  ///   - validation: Validation samples used for periodic generation checks.
  public func fit(_ data: [DatasetModel], _ validation: [DatasetModel]) {
    let batches = data.batched(into: batchSize)
    let _ = validation.batched(into: batchSize)
    
    let valBatches = validation.batched(into: batchSize)

    //epochs
    var trainingBatches: [(data: [Tensor], labels: [Tensor])] = []
    batches.forEach { b in
      if b.count == batchSize {
        let trainingSet = splitDataset(b)
        trainingBatches.append(trainingSet)
      }
    }

    var validationBatches: [(data: [Tensor], labels: [Tensor])] = []
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
  
  /// Generates one synthetic sample from random noise.
  ///
  /// - Returns: Generated tensor detached from the training graph.
  public func generate() -> Tensor {
    self.generator.isTraining = false
    let out = generator([noise()])
    return out.first?.detached() ?? Tensor()
  }
  
  /// Runs discriminator inference on input tensors.
  ///
  /// - Parameter input: Tensors to score as real/fake.
  /// - Returns: First discriminator output tensor.
  public func discriminate(_ input: [Tensor]) -> Tensor {
    let out = discriminator(input)
    return out.first ?? Tensor()
  }
  
  @discardableResult
  /// Exports discriminator and generator models.
  ///
  /// - Parameters:
  ///   - overrite: When `false`, appends timestamps to filenames.
  ///   - compress: When `true`, writes compact JSON.
  /// - Returns: Tuple of optional URLs for discriminator and generator models.
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
  
  internal func discriminatorStep(_ real: [Tensor], labels: [Tensor]) {
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
    
    // this only gets the gradients wrt to this generated data, which stops at the input of the discriminator
    let output = trainOn(generatedData.data,
                         labels: generatedData.labels,
                         wrt: generatedData.wrt)
    
    let loss = output.loss
    generator.metricsReporter?.update(metric: .generatorLoss, value: loss)
    
    generator.apply(output.gradients)
    generator.step()
  }
  
  internal func trainOn(_ batch: [Tensor],
                        labels: [Tensor],
                        wrt: TensorBatch? = nil,
                        requiresGradients: Bool = true) -> Optimizer.Output {
    discriminator.fit(batch,
                      labels: labels,
                      wrt: wrt,
                      lossFunction: lossFunction,
                      validation: false,
                      requiresGradients: requiresGradients)
  }
  
  internal func getGenerated(_ label: TrainingType, detatch: Bool = false, count: Int) -> (data: [Tensor],
                                                                                           labels: [Tensor],
                                                                                           wrt: TensorBatch) {
    var fakeData: [Tensor] = [Tensor](repeating: Tensor(), count: count)
    var fakeLabels: [Tensor] = [Tensor](repeating: Tensor(), count: count)
    var wrt: TensorBatch = [Tensor](repeating: Tensor(), count: count)

    Array(0..<count).concurrentForEach(workers: Constants.maxWorkers) { _, index in
      
      let noise = self.noise()
      
      var sample = self.generator([noise])[safe: 0, Tensor()]
      
      if detatch {
        sample = sample.detached()
      }
      
      let localLabelValue = label == .fake ? self.fakeLabel : self.realLabel
      var dataLabel = Tensor([localLabelValue])
      //assuming the label is 1.0 or greater
      //we need to reverse if label is <= 0
      if let noise = self.discriminatorNoiseFactor, noise > 0, noise < 1 {
        //cap factor between 0 and 1
        let factor = min(1.0, max(0.0, noise))
        let min = min(localLabelValue, abs(localLabelValue - factor))
        let max = max(localLabelValue, abs(localLabelValue - factor))
        dataLabel = Tensor([Tensor.Scalar.random(in: (min...max))])
      }
      
      fakeData[index] = sample
      fakeLabels[index] = dataLabel
      wrt[index] = noise
    }

    return (fakeData, fakeLabels, wrt)
  }

  private func splitDataset(_ data: [DatasetModel]) -> (data: [Tensor], labels: [Tensor]) {
    var labels: [Tensor] = []
    var input: [Tensor] = []
    
    data.forEach { d in
      labels.append(d.label)
      input.append(d.data)
    }
    
    return (input, labels)
  }
}
