//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/11/22.
//

import Foundation
import NumSwift

internal protocol GANTrainingDataBuilder {
  var generator: Brain? { get set }
  var discriminator: Brain? { get set }
  var batchSize: Int { get set }
  var lossFunction: GANLossFunction { get set }
  var discriminatorNoiseFactor: Float? { get set }
  
  func getRandomBatch(data: [TrainingData]) -> [TrainingData]
  func getGeneratedData(type: GANTrainingType,
                        noise: [Float]) -> [TrainingData]
  func getInterpolated(real: TrainingData, fake: TrainingData) -> TrainingData
}

internal extension GANTrainingDataBuilder {
  
  func getInterpolated(real: TrainingData,
                       fake: TrainingData) -> TrainingData {
    let epsilon = Float.random(in: 0...1)
    
    let realNew = real.data
    let fakeNew = fake.data
    
    let inter = (realNew * epsilon) + (fakeNew * (1 - epsilon))
    
    let interTrainingData = TrainingData(data: inter, correct: [1.0])
    
    return interTrainingData
  }
  
  func getRandomBatch(data: [TrainingData]) -> [TrainingData] {
    var newData: [TrainingData] = []
    for _ in 0..<batchSize {
      if let element = data.randomElement() {
        newData.append(element)
      }
    }
    return newData
  }
  
  func getGeneratedData(type: GANTrainingType,
                        noise: [Float]) -> [TrainingData] {
    var fakeData: [TrainingData] = []
    guard let gen = generator else {
      return []
    }
    
    for _ in 0..<batchSize {
      let sample = gen.feed(input: noise)
      
      let label = lossFunction.label(type: type)
      
      var training = TrainingData(data: sample, correct: [label])
      //assuming the label is 1.0 or greater
      //we need to reverse if label is <= 0
      if let noise = discriminatorNoiseFactor, noise > 0, noise < 1 {
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

}
