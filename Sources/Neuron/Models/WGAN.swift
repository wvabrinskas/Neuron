//
//  File.swift
//  
//
//  Created by William Vabrinskas on 7/8/22.
//

import Foundation
import NumSwift

public class WGAN<N: TensorNumeric>: GAN<N> {
  public override var realLabel: Tensor<N>.Scalar { -1.0 }
  public override var fakeLabel: Tensor<N>.Scalar { 1.0 }
  
  public override var lossFunction: LossFunction { .wasserstein  }
  
  override func discriminatorStep(_ real: [Tensor<N>], labels: [Tensor<N>]) {
    discriminator.zeroGradients()
    
    let fake = getGenerated(.fake, detatch: true, count: batchSize)
    let fakeOutput = trainOn(fake.data, labels: fake.labels)

    let realOutput = trainOn(real, labels: labels)
    
    discriminator.apply(fakeOutput.gradients + realOutput.gradients)
    
    let realLoss = realOutput.loss
    let fakeLoss = fakeOutput.loss
    
    let totalSumLoss = (fakeLoss + realLoss)
          
    discriminator.metricsReporter?.update(metric: .criticLoss, value: totalSumLoss)
    discriminator.metricsReporter?.update(metric: .realImageLoss, value: realLoss)
    discriminator.metricsReporter?.update(metric: .fakeImageLoss, value: fakeLoss)
    
    discriminator.step()
  }
  
  override func generatorStep() {
    generator.zeroGradients()
    //setting them to `.real` will set labels to -1 getting us the negative mean
    let fake = getGenerated(.real, count: batchSize)
    
    let fakeOut = trainOn(fake.data, labels: fake.labels)
    let fakeGradients = fakeOut.gradients
    
    generator.apply(fakeGradients)
    generator.step()
    
    generator.metricsReporter?.update(metric: .generatorLoss, value: fakeOut.loss)
  }

}
