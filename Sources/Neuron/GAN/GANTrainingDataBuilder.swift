//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/11/22.
//

import Foundation

internal protocol GANTrainingDataBuilder {
  var generator: Brain? { get set }
  var discriminator: Brain? { get set }
  var batchSize: Int { get set }
  var lossFunction: GANLossFunction { get set }
  var discriminatorNoiseFactor: Float? { get set }
  
  func getRandomBatch(data: [TrainingData]) -> [TrainingData]
  func getGeneratedData(type: GANTrainingType,
                        noise: [Float],
                        size: Int) -> [TrainingData]
}

internal extension GANTrainingDataBuilder {
  func getRandomBatch(data: [TrainingData]) -> [TrainingData] {
    var newData: [TrainingData] = []
    for _ in 0..<self.batchSize {
      if let element = data.randomElement() {
        newData.append(element)
      }
    }
    return newData
  }
  
  func getGeneratedData(type: GANTrainingType,
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

}
