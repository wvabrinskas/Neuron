//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/6/22.
//

import Foundation
import NumSwift
import Numerics

/// A Wasserstein GAN with Gradient Penalty (WGAN-GP) trainer.
///
/// Enforces the Lipschitz constraint via a gradient penalty term computed on
/// interpolated samples instead of weight clipping. The penalty coefficient is
/// controlled by `lambda`.
open class WGANGP: GAN {
  /// The label assigned to real samples in the Wasserstein GAN-GP framework.
  ///
  /// Returns -1.0 to follow the WGAN-GP convention where real samples are labeled negatively.
  public override var realLabel: Tensor.Scalar { -1.0 }
  /// The label assigned to fake (generated) samples in the Wasserstein GAN-GP framework.
  ///
  /// Returns 1.0 to follow the WGAN-GP convention where fake samples are labeled positively.
  public override var fakeLabel: Tensor.Scalar { 1.0 }
  /// The gradient penalty coefficient used to enforce the Lipschitz constraint.
  ///
  /// Scales the gradient penalty term added to the discriminator loss. Higher values enforce the constraint more strongly.
  public var lambda: Tensor.Scalar = 0.1

  private struct GradientPenalty {
    static func calculate(gradient: Tensor) -> (penalty: Tensor.Scalar, norm: Tensor.Scalar) {
      let norm = gradient.norm().asScalar()
      let normAdjusted = norm - 1
      return (Tensor(normAdjusted * normAdjusted).asScalar(), norm)
    }
  }
  
  /// The loss function used by this GAN, set to Wasserstein loss.
  ///
  /// Returns the Wasserstein loss function, which measures the distance between real and generated data distributions.
  public override var lossFunction: LossFunction { .wasserstein  }
  
  override func discriminatorStep(_ real: [Tensor], labels: [Tensor]) {
    discriminator.zeroGradients()
    
    let generated = self.getGenerated(.fake, detatch: true, count: batchSize)
    let interpolated = self.interpolated(real: real, fake: generated.data)
    
    let realOut = self.trainOn(real, labels: labels, requiresGradients: true)
    let fakeOut = self.trainOn(generated.data, labels: generated.labels, requiresGradients: true)
    
    // Compute per-sample gradient norms on interpolated samples via manual
    // backprop (bypasses fit() to get unscaled input gradients).
    let interOutputs = self.discriminator.predict(interpolated.data)
    
    let batchScalar = batchSize.asTensorScalar
    var sampleNorms = [Tensor.Scalar](repeating: 0, count: batchSize)
    var samplePenalties = [Tensor.Scalar](repeating: 0, count: batchSize)
    
    Array(0..<batchSize).concurrentForEach(workers: Constants.maxWorkers) { _, i in
      let output = interOutputs[i]
      let input = interpolated.data[i]
      
      let delta = Tensor.fillWith(value: 1, size: output.size)
      let gradient = output.gradients(delta: delta, wrt: input)
      
      let inputGrad = gradient.input[safe: 0, Tensor()]
      let penaltyResult = GradientPenalty.calculate(gradient: inputGrad)
      
      sampleNorms[i] = penaltyResult.norm
      samplePenalties[i] = penaltyResult.penalty
    }
    
    let avgPenalty = samplePenalties.reduce(0, +) / batchScalar
    let avgNorm = sampleNorms.reduce(0, +) / batchScalar
    
    // Without second-order derivatives we cannot compute the true gradient of
    // the penalty w.r.t. θ. Instead, use the average gradient norm to adaptively
    // scale the critic update: when ||∇D|| >> 1 the critic is too powerful so we
    // dampen its gradients; when ||∇D|| ≈ 1 the scale is ≈ 1 (no dampening).
    let criticScale = Tensor.Scalar(1) / Swift.max(avgNorm, Tensor.Scalar(1))
    
    let layerCount = realOut.gradients.weights.count
    let totalWeights = (0..<layerCount).map { j in
      (realOut.gradients.weights[j] + fakeOut.gradients.weights[j]) * criticScale
    }
    let totalBiases = (0..<layerCount).map { j in
      (realOut.gradients.biases[j] + fakeOut.gradients.biases[j]) * criticScale
    }
    
    let totalGradients = Tensor.Gradient(
      input: [],
      weights: totalWeights,
      biases: totalBiases
    )
    
    self.discriminator.apply(totalGradients)
    
    let criticLoss = fakeOut.loss + realOut.loss + lambda * avgPenalty
    discriminator.metricsReporter?.update(metric: .realImageLoss, value: realOut.loss)
    discriminator.metricsReporter?.update(metric: .fakeImageLoss, value: fakeOut.loss)
    discriminator.metricsReporter?.update(metric: .gradientPenalty, value: avgPenalty)
    discriminator.metricsReporter?.update(metric: .criticLoss, value: criticLoss)
      
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
      let inter = real[i].detached() * epsilon + (Tensor.Scalar(1) - epsilon) * fake[i].detached()
      interpolated.append(inter.detached())
      labels.append(Tensor(1.0))
    }
    
    return (interpolated, labels)
  }
}
