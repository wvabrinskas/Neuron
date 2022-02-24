//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/23/22.
//

import Foundation
import NumSwift
import Combine

public class MNIST {
  @Published public var trainingData: [ConvTrainingData] = []
  @Published public var validationData: [ConvTrainingData] = []
  private let mnistSize: TensorSize = (28,28,1)
  public let loadQueue: DispatchQueue = DispatchQueue(label: "mnist.load.queue.neuron",
                                                       qos: .default,
                                                       attributes: .concurrent,
                                                       autoreleaseFrequency: .inherit,
                                                       target: .global())

  public enum MNISTType: String, CaseIterable {
    case trainingSet = "train-images"
    case trainingLabels = "train-labels"
    case valSet = "t10k-images"
    case valLabels = "t10k-labels"
    
    var startingByte: UInt8 {
      switch self {
      case .trainingSet:
        return 0x0016
      case .trainingLabels:
        return 0x0008
      case .valSet:
        return 0x0016
      case .valLabels:
        return 0x0008
      }
    }
    
    var shape: TensorSize {
      switch self {
      case .trainingSet, .valSet:
        return (28,28,1)
      case .trainingLabels, .valLabels:
        return (1,1,1)
      }
    }
  }
  
  public func build() {
    print("Loading MNIST dataset into memory. This could take a while")
    DispatchQueue.global().async { [weak self] in
      guard let sSelf = self else {
        return
      }
      
      let trainingData = sSelf.get(type: .trainingSet)
      let trainingLabels = sSelf.get(type: .trainingLabels)
      let valData = sSelf.get(type: .valSet)
      let valLabels = sSelf.get(type: .valLabels)
      
      var trainingDataWithLabels: [ConvTrainingData] = []
      for t in 0..<trainingData.count {
        let data = trainingData[t]
        let tLabel = trainingLabels[t].flatMap { $0 }.first ?? 0
        let label = sSelf.buildLabel(value: Int(tLabel))
        let object = ConvTrainingData(data: [data], label: label)
        trainingDataWithLabels.append(object)
      }
      
      sSelf.trainingData = trainingDataWithLabels
      
      var valDataWithLabels: [ConvTrainingData] = []
      for t in 0..<valData.count {
        let data = valData[t]
        let tLabel = valLabels[t].flatMap { $0 }.first ?? 0
        let label = sSelf.buildLabel(value: Int(tLabel))
        let object = ConvTrainingData(data: [data], label: label)
        valDataWithLabels.append(object)
      }
      
      sSelf.validationData = valDataWithLabels
    }

  }
  
  private func buildLabel(value: Int) -> [Float] {
    guard value > 0 else {
      return []
    }
    var labels = [Float].init(repeating: 0, count: 10)
    labels[value] = 1
    return labels
  }
  
  public func get(type: MNISTType) -> [[[Float]]] {
    let trainingImages = Bundle.module.path(forResource: type.rawValue, ofType: nil)
    
    guard let trainingImages = trainingImages else {
      return []
    }

    let url = URL(fileURLWithPath: trainingImages)
    
    do {
      let data = try Data(contentsOf: url)
      let array = data.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [UInt8] in
        return Array<UInt8>(pointer[Int(type.startingByte)..<pointer.count])
      }
      
      let t = array.map { Float($0) }
        
      return t.reshape(columns: type.shape.columns).batched(into: type.shape.rows)
    } catch {
      print("error")
      return []
    }
  }
}
