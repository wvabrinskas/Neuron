//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/24/22.
//

import Foundation
import Combine

//public typealias DatasetModel = [(data: Tensor, label: Tensor)]

public struct DatasetModel: Equatable {
  public var data: Tensor
  public var label: Tensor
  
  public init(data: Tensor, label: Tensor) {
    self.data = data
    self.label = label
  }
}

public typealias DatasetData = (training: [DatasetModel], val: [DatasetModel])

public protocol Dataset {

  var data: DatasetData { get set }
  var complete: Bool { get set }
  var dataPassthroughSubject: PassthroughSubject<DatasetData, Never> { get }
  var dataPublisher: AnyPublisher<DatasetData, Never> { get }
  
  func read<T: FloatingPoint>(path: String, offset: Int, scaleBy: T) -> [T]
  func read<T: FloatingPoint>(data: Data, offset: Int, scaleBy: T) -> [T]
  func bitmap(path: String, offset: Int) -> [UInt8]
  func format(data: Data, offset: Int) -> [UInt8]
  
  func build() async -> DatasetData
  func build()
}

public extension Dataset {
  var dataPublisher:  AnyPublisher<DatasetData, Never> {
    dataPassthroughSubject.eraseToAnyPublisher()
  }
  
  func format(data: Data, offset: Int) -> [UInt8] {
    let array = data.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [UInt8] in
      return Array<UInt8>(pointer[offset..<pointer.count])
    }
    
    return array
  }
  
  func bitmap(path: String, offset: Int) -> [UInt8] {
    let url = URL(fileURLWithPath: path)
    
    do {
      let data = try Data(contentsOf: url)
      return format(data: data, offset: offset)
    } catch {
      print(error.localizedDescription)
      return []
    }
  }
  
  func read<T: FloatingPoint>(data: Data, offset: Int, scaleBy: T) -> [T] {
    let bitmap = format(data: data, offset: offset)
    let result = bitmap.map { T($0) / scaleBy }
    return result
  }
  
  func read<T: FloatingPoint>(path: String, offset: Int, scaleBy: T) -> [T] {
    let bitmap = bitmap(path: path, offset: offset)
    let result = bitmap.map { T($0) / scaleBy }
    return result
  }
}
