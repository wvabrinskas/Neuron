//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/13/22.
//

import Foundation
import NumSwift

public class WGANGP: GAN {
  public override init(generator: Brain? = nil,
                       discriminator: Brain? = nil,
                       epochs: Int,
                       criticTrainPerEpoch: Int = 5,
                       gradientPenaltyLambda: Float = 10,
                       batchSize: Int,
                       metrics: Set<Metric> = []) {
    
    super.init(generator: generator,
               discriminator: discriminator,
               epochs: epochs,
               criticTrainPerEpoch: criticTrainPerEpoch,
               gradientPenaltyLambda: gradientPenaltyLambda,
               batchSize: batchSize,
               metrics: metrics)
    
    self.lossFunction = .wasserstein
  }
  
  internal override func criticStep(real: [TrainingData], fake: [TrainingData]) -> Float {
    guard let dis = discriminator,
          real.count == fake.count else {
      return 0
    }
    
    defer {
      discriminator?.zeroGradients()
    }
    
    var realLossAverage: Float = 0
    var fakeLossAverage: Float = 0
    var penaltyAverage: Float = 0
    
    for i in 0..<real.count {
      dis.zeroGradients()
      
      let realSample = real[i]
      let fakeSample = fake[i]
      let interSample = getInterpolated(real: realSample, fake: fakeSample)
      
      let realLoss = batchDiscriminate([realSample], type: .real).loss
      let fakeLoss = batchDiscriminate([fakeSample], type: .fake).loss
    
      let interLoss = batchDiscriminate([interSample], type: .real).loss
      
      dis.backpropagate(with: [interLoss])
      
      if let networkGradients = dis.gradients()[safe: 1]?.flatMap({ $0 }) {
        let penalty = GradientPenalty.calculate(gradient: networkGradients)
        penaltyAverage += penalty / Float(real.count)
      }
    
      realLossAverage += realLoss / Float(real.count)
      fakeLossAverage += fakeLoss / Float(fake.count)
    }
    
    let penalty = gradientPenaltyLambda * penaltyAverage
    gradientPenalty = penalty
    
    let criticLoss = fakeLossAverage - realLossAverage + penalty
    return criticLoss
  }
  
  internal override func generatorStep(fake: [TrainingData]) -> Float {
    let genOutput = self.batchDiscriminate(fake, type: .real)
    return -1 * genOutput.loss
  }
}
