//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/6/22.
//

import Foundation
import NumSwift
import Numerics

open class WGANGP: GAN {
  public override var realLabel: Tensor.Scalar { -1.0 }
  public override var fakeLabel: Tensor.Scalar { 1.0 }
  public var lambda: Tensor.Scalar = 0.1

  private struct GradientPenalty {
    static func calculate(gradient: Tensor) -> Tensor {
      let sumOfSquares = gradient.sumOfSquares().asScalar()
      let norm = Tensor.Scalar.sqrt(sumOfSquares + .stabilityFactor) - 1
      return Tensor(norm * norm)
    }
    
    static func calculate(gradients: [Tensor]) -> [Tensor] {
      gradients.map { calculate(gradient: $0) }
    }
  }
  
  public override var lossFunction: LossFunction { .wasserstein  }
  
  override func discriminatorStep(_ real: [Tensor], labels: [Tensor]) {
    discriminator.zeroGradients()
    
    var avgPenalty: Tensor.Scalar = 0
    var avgCriticLoss: Tensor.Scalar = 0
    var avgRealLoss: Tensor.Scalar = 0
    var avgFakeLoss: Tensor.Scalar = 0
    
    Array(0..<batchSize).concurrentForEach(workers: Constants.maxWorkers) { _, i in
      // create data
      let realSample = real[i]
      let realLabel = labels[i]
      
      let generated = self.getGenerated(.fake, detatch: true, count: 1)
      let interpolated = self.interpolated(real: [realSample], fake: generated.data)
      
      let fakeSample = generated.data[safe: 0, Tensor()]
      let fakeLabel = generated.labels[safe: 0, Tensor()]
      
      let interSample = interpolated.data[safe: 0, Tensor()]
      let interLabel = interpolated.labels[safe: 0, Tensor()]

      // get real output
      let realOut = self.trainOn([realSample], labels: [realLabel], requiresGradients: true)
      let realOutput = realOut.outputs[safe: 0, Tensor()]
      
      // get fake output
      let fakeOut = self.trainOn([fakeSample], labels: [fakeLabel], requiresGradients: true)
      let fakeOutput = fakeOut.outputs[safe: 0, Tensor()]

      // get gradient for interpolated
      let interOut = self.trainOn([interSample], labels: [interLabel], requiresGradients: true)
      // interpolated gradients wrt to interpolated sample
      let interGradients = interOut.gradients.input[safe: 0, Tensor()]
      
      // calculate gradient penalty
      let normGradients = interGradients.norm().asScalar()
      let penalty = LossFunction.meanSquareError.calculate([normGradients], correct: [1.0]) // just using this for the calculation part
            
      // calculate critic loss vs real and fake.
      let criticCost = fakeOutput - realOutput
      let criticLoss = criticCost + self.lambda * penalty
          
      let part1 = Tensor.Scalar(2 / fakeOutput.size.depth) * self.lambda
      let part2 = normGradients - 1
      let part3 = interOut.outputs[safe: 0, Tensor()] / normGradients
      let dGradInter = part1 * part2 * part3
      
      let interpolatedGradients = interOut.outputs[safe: 0, Tensor()].gradients(delta: dGradInter,
                                                                                wrt: interSample)

      let totalGradients = fakeOut.gradients + realOut.gradients + interpolatedGradients
      
      self.discriminator.apply(totalGradients)

      avgRealLoss += realOut.loss / self.batchSize.asTensorScalar
      avgFakeLoss += fakeOut.loss / self.batchSize.asTensorScalar
      avgPenalty += penalty / self.batchSize.asTensorScalar
      avgCriticLoss += criticLoss.asScalar() / self.batchSize.asTensorScalar
    }
  
    discriminator.metricsReporter?.update(metric: .realImageLoss, value: avgRealLoss)
    discriminator.metricsReporter?.update(metric: .fakeImageLoss, value: avgFakeLoss)
    discriminator.metricsReporter?.update(metric: .gradientPenalty, value: avgPenalty)
    discriminator.metricsReporter?.update(metric: .criticLoss, value: avgCriticLoss)
      
    discriminator.step()
  }
  
  override func generatorStep() {
    generator.zeroGradients()
    let fake = getGenerated(.real, count: batchSize)
        
    let fakeOut = trainOn(fake.data,
                          labels: fake.labels,
                          wrt: fake.wrt)
    
    // these gradients have the discriminator gradients as well. Maybe we can speed it up by removing them somehow?
    generator.apply(fakeOut.gradients)
    generator.step()
    
    generator.metricsReporter?.update(metric: .generatorLoss, value: fakeOut.loss)
  }
  
  internal func interpolated(real: [Tensor], fake: [Tensor]) -> (data: [Tensor], labels: [Tensor]) {
    var interpolated: [Tensor] = []
    var labels: [Tensor] = []
    
    for i in 0..<real.count {
      let epsilon = Tensor.Scalar.random(in: 0...1)
      let inter = real[i] * epsilon + (Tensor.Scalar(1) - epsilon) * fake[i]
      interpolated.append(inter)
      labels.append(Tensor(1.0))
    }
    
    return (interpolated, labels)
  }
}
