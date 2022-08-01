//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/6/22.
//

import Foundation
import NumSwift

public class WGANGP: GAN {
  public override var realLabel: Tensor.Scalar { -1.0 }
  public override var fakeLabel: Tensor.Scalar { 1.0 }
  public var lambda: Tensor.Scalar = 0.1

  private struct GradientPenalty {
    static func calculate(gradient: Tensor) -> Tensor {
      let norm = sqrt(gradient.value.sumOfSquares + 1e-12) - 1
      return Tensor(pow(norm, 2))
    }
    
    static func calculate(gradients: [Tensor]) -> [Tensor] {
      gradients.map { calculate(gradient: $0) }
    }
  }
  
  public override var lossFunction: LossFunction { .wasserstein  }
  
  // SLOW =(
  func NEW_discriminatorStep(_ real: [Tensor], labels: [Tensor]) {
    discriminator.zeroGradients()
    
    let realOutput = trainOn(real, labels: labels)
    discriminator.apply(realOutput.gradients)
    
    let fake = getGenerated(.fake, detatch: true, count: batchSize)
    let fakeOutput = trainOn(fake.data, labels: fake.labels)
    discriminator.apply(fakeOutput.gradients)

    let interpolated = self.interpolated(real: real, fake: fake.data)
    
    let interOut = trainOn(interpolated.data, labels: interpolated.labels)
    let penalty = interOut.gradients.input.map { Tensor(LossFunction.meanSquareError.calculate([$0.norm().asScalar()], correct: [1.0])) }
    
    let interAccumulator = GradientAccumulator()
    interOut.outputs.gradients(penalty.map { Tensor(LossFunction.meanSquareError.derivative([self.lambda * $0.asScalar()], correct: [1.0]))}).forEach { interAccumulator.insert($0) }
    discriminator.apply(interAccumulator.accumulate(clearAtEnd: true))
    
    let realLoss = realOutput.loss
    let fakeLoss = fakeOutput.loss
              
    // calculate critic loss vs real and fake.
    // Real is already multiplied by -1 due to the label, so we can just add them
    let criticCost = realLoss + fakeLoss
    let gp = penalty.mean * self.lambda
    let criticLoss = criticCost + gp.asScalar()

    discriminator.metricsReporter?.update(metric: .criticLoss, value: criticLoss)
    discriminator.metricsReporter?.update(metric: .realImageLoss, value: realLoss)
    discriminator.metricsReporter?.update(metric: .fakeImageLoss, value: fakeLoss)
    
    discriminator.step()
  }
  
  override func discriminatorStep(_ real: [Tensor], labels: [Tensor]) {
    discriminator.zeroGradients()
    
    var avgPenalty: Tensor.Scalar = 0
    var avgCriticLoss: Tensor.Scalar = 0
    var avgRealLoss: Tensor.Scalar = 0
    var avgFakeLoss: Tensor.Scalar = 0
    
    let generated = self.getGenerated(.fake, detatch: true, count: batchSize)
    let interpolated = self.interpolated(real: real, fake: generated.data)

    let workers = min(16, Int(ceil(batchSize.asTensorScalar / 4)))
    Array(0..<batchSize).concurrentForEach(workers: workers) { _, i in
      // create data
      let realSample = real[i]
      let realLabel = labels[i]
      
      let fakeSample = generated.data[i]
      let fakeLabel = generated.labels[i]
      
      let interSample = interpolated.data[i]
      let interLabel = interpolated.labels[i]

      // get real output
      let realOut = self.trainOn([realSample], labels: [realLabel], requiresGradients: true)
      let realOutput = realOut.outputs[safe: 0, Tensor()]
      
      // get fake output
      let fakeOut = self.trainOn([fakeSample], labels: [fakeLabel], requiresGradients: true)
      let fakeOutput = fakeOut.outputs[safe: 0, Tensor()]

      // get gradient for interpolated
      let interOut = self.trainOn([interSample], labels: [interLabel], requiresGradients: true)
      let interGradients = interOut.gradients.input[safe: 0, Tensor()]
      let normGradients = interGradients.norm().asScalar()
      let penalty = LossFunction.meanSquareError.calculate([normGradients], correct: [1.0])
            
      // calculate critic loss vs real and fake.
      // Real is already multiplied by -1 due to the label, so we can just add them
      let criticCost = fakeOutput - realOutput
      let gp = penalty * self.lambda
      let criticLoss = criticCost + gp
      
      // calculate gradients w.r.t to the interpolated gradients
      let derivativeSumSqr = (interGradients * 2).sum()
     // let derivOfGrads = self.discriminator.predict([derivativeSumSqr])[safe: 0, Tensor()].asScalar()
      let derivativeLoss = Tensor(self.lambda * (normGradients - 1) * pow(interGradients.sumOfSquares().asScalar(), -0.5) * derivativeSumSqr)
      
      let interpolatedGradients = interOut.outputs[safe: 0, Tensor()].gradients(delta: derivativeLoss)
      
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
    let fake = getGenerated(.fake, count: batchSize)
    
    // invert labels so that we can get the negative mean
    let labels = fake.labels.map { $0 * -1 }
    
    let fakeOut = trainOn(fake.data, labels: labels)
    let fakeGradients = fakeOut.gradients
    
    generator.apply(fakeGradients)
    generator.step()
    
    generator.metricsReporter?.update(metric: .generatorLoss, value: fakeOut.loss)
  }
  
  internal func interpolated(real: [Tensor], fake: [Tensor]) -> (data: [Tensor], labels: [Tensor]) {
    var interpolated: [Tensor] = []
    var labels: [Tensor] = []
    
    for i in 0..<real.count {
      let epsilon = Tensor.Scalar.random(in: 0...1)
      let realImage = real[i].value
      let fakeImage = fake[i].value

      let inter = (realImage * epsilon) + (( 1 - epsilon) * fakeImage)
      interpolated.append(Tensor(inter))
      labels.append(Tensor(1.0))
    }
    
    return (interpolated, labels)
  }
}
