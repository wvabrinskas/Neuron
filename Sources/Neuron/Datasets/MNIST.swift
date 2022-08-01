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
  
  private let mnistSize: [Int] = [28,28,1]
  private var numToGet: Int?
  private var zeroCentered: Bool
  private var correctLabel: [Float] = []

  public enum MNISTType: String, CaseIterable {
    case trainingSet = "train-images"
    case trainingLabels = "train-labels"
    case valSet = "t10k-images"
    case valLabels = "t10k-labels"
    
    var startingByte: UInt8 {
      switch self {
      case .trainingSet:
        return 0x0013
      case .trainingLabels:
        return 0x0008
      case .valSet:
        return 0x0013
      case .valLabels:
        return 0x0008
      }
    }
    
    var shape: [Int] {
      switch self {
      case .trainingSet, .valSet:
        return [28,28,1]
      case .trainingLabels, .valLabels:
        return [1,1,1]
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
  
  public init(only num: Int? = nil,
              label: [Float] = [],
              zeroCentered: Bool = false) {
    if let num = num {
      self.numToGet = num
    }
    self.correctLabel = label
    self.zeroCentered = zeroCentered
  }
  
  public func build(only num: Int) async -> DatasetData {
    self.numToGet = num
    return await build()
  }
  
  /// Build with support for Combine
  public func build() {
    Task {
      await build()
    }
  }
  
  /// Build with async/await support
  /// - Returns: downloaded dataset
  public func build() async -> DatasetData {
    guard complete == false else {
      print("MNIST has already been loaded")
      return self.data
    }
    
    print("Loading MNIST dataset into memory. This could take a while")
    
    
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
      
      var trainingDataWithLabels: [DatasetModel] = []
      var validationDataWithLabels: [DatasetModel] = []
      
      let validationData = valDataSets.first { $0.type == .valSet }?.data ?? []
      let validationLabels = valDataSets.first { $0.type == .valLabels }?.data ?? []
      
      let trainingData = trainingDataSets.first { $0.type == .trainingSet }?.data ?? []
      let trainingLabels = trainingDataSets.first { $0.type == .trainingLabels }?.data ?? []
      
      for i in 0..<trainingData.count {
        let tD = trainingData[i]
        let tL = trainingLabels[i].first?.first ?? -1
        if numToGet == nil || Int(tL) == numToGet {
          let conv = DatasetModel(data: Tensor(tD), label: Tensor(self.buildLabel(value: Int(tL)))) //only one channel for MNIST
          trainingDataWithLabels.append(conv)
        }
      }
      
      for i in 0..<validationData.count {
        let tD = validationData[i]
        let tL = validationLabels[i].first?.first ?? -1
        if numToGet == nil || Int(tL) == numToGet {
          let conv = DatasetModel(data: Tensor(tD), label: Tensor(self.buildLabel(value: Int(tL)))) //only one channel for MNIST
          validationDataWithLabels.append(conv)
        }
      }
      
      return (trainingDataWithLabels, validationDataWithLabels)
    })
    
    return data
  }
  
  private func buildLabel(value: Int) -> [Float] {
    if !self.correctLabel.isEmpty {
      return correctLabel
    }
    
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
    
    let shouldZeroCenter: Bool = zeroCentered && (type == .valSet || type == .trainingSet)
    
    var scale: Float = shouldZeroCenter ? 1 : 255
    
    if type == .trainingLabels || type == .valLabels {
      scale = type.modifier
    }
    
    var result = read(path: path, offset: Int(type.startingByte), scaleBy: scale)
    
    if shouldZeroCenter {
      result = result.map { ($0 - 127.5) / 127.5 }
    }
    
    let columns = type.shape[safe: 0] ?? 0
    let rows = type.shape[safe: 1] ?? 0
    
    let shaped = result.reshape(columns: columns).batched(into: rows)
    
    return shaped
  }
}
