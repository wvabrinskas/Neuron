//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/23/22.
//

import Foundation
import NumSwift
import Combine
 
public class MNIST: Dataset {
  public var data: DatasetData = ([], []) {
    didSet {
      dataPassthroughSubject.send(data)
    }
  }
  public var complete: Bool = false
  public let dataPassthroughSubject = PassthroughSubject<DatasetData, Never>()
  
  private let mnistSize: TensorSize = (28,28,1)
  
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
    
    var modifier: Float {
      switch self {
      case .trainingSet, .valSet:
        return 255.0
      case .valLabels, .trainingLabels:
        return 1.0
      }
    }
  }
  
  public func build() {
    guard complete == false else {
      print("MNIST has already been loaded")
      return
    }
    print("Loading MNIST dataset into memory. This could take a while")

    Task {
      self.data = await withTaskGroup(of: (data: [[[Float]]], type: MNISTType).self, body: { group in
        
        var trainingDataSets: [(data: [[[Float]]], type: MNISTType)] = []
        trainingDataSets.reserveCapacity(2)

        var valDataSets: [(data: [[[Float]]], type: MNISTType)] = []
        valDataSets.reserveCapacity(2)
        
        group.addTask(priority: .userInitiated) {
          let trainingData = self.get(type: .trainingSet)
          return (trainingData, .trainingSet)
        }
        
        group.addTask(priority: .userInitiated) {
          let trainingData = self.get(type: .trainingLabels)
          return (trainingData, .trainingLabels)
        }
        
        group.addTask(priority: .userInitiated) {
          let trainingData = self.get(type: .valSet)
          return (trainingData, .valSet)
        }
        
        group.addTask(priority: .userInitiated) {
          let trainingData = self.get(type: .valLabels)
          return (trainingData, .valLabels)
        }
        
        for await data in group {
          if data.type == .trainingLabels || data.type == .trainingSet {
            trainingDataSets.append(data)
          } else if data.type == .valLabels || data.type == .valSet {
            valDataSets.append(data)
          }
        }
          
        var trainingDataWithLabels: [ConvTrainingData] = []
        var validationDataWithLabels: [ConvTrainingData] = []
        
        let validationData = valDataSets.first { $0.type == .valSet }?.data ?? []
        let validationLabels = valDataSets.first { $0.type == .valLabels }?.data ?? []

        let trainingData = trainingDataSets.first { $0.type == .trainingSet }?.data ?? []
        let trainingLabels = trainingDataSets.first { $0.type == .trainingLabels }?.data ?? []
        
        for i in 0..<trainingData.count {
          let tD = trainingData[i]
          let tL = trainingLabels[i].first?.first ?? -1
          let conv = ConvTrainingData(data: [tD], label: self.buildLabel(value: Int(tL))) //only one channel for MNIST
          trainingDataWithLabels.append(conv)
        }
        
        for i in 0..<validationData.count {
          let tD = validationData[i]
          let tL = validationLabels[i].first?.first ?? -1
          let conv = ConvTrainingData(data: [tD], label: self.buildLabel(value: Int(tL))) //only one channel for MNIST
          validationDataWithLabels.append(conv)
        }
        
        return (trainingDataWithLabels, validationDataWithLabels)
      })
      
    }
  }
  
  private func buildLabel(value: Int) -> [Float] {
    guard value >= 0 else {
      return []
    }
    var labels = [Float].init(repeating: 0, count: 10)
    labels[value] = 1
    return labels
  }
  
  public func get(type: MNISTType) -> [[[Float]]] {
    let path = Bundle.module.path(forResource: type.rawValue, ofType: nil)
    
    guard let path = path else {
      return []
    }
    
    let t = read(path: path, offset: Int(type.startingByte), scaleBy: type.modifier)
    
    return t.reshape(columns: type.shape.columns).batched(into: type.shape.rows)
  }
}
