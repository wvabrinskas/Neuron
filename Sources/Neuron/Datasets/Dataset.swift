//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/24/22.
//

import Foundation
import Combine

public typealias DatasetData = (training: [ConvTrainingData], val: [ConvTrainingData])

public protocol Dataset {
  var data: DatasetData { get set }
  var complete: Bool { get set }
  var dataPassthroughSubject: PassthroughSubject<DatasetData, Never> { get }
  var dataPublisher: AnyPublisher<DatasetData, Never> { get }
  
  func read<T: FloatingPoint>(path: String, offset: Int, scaleBy: T) -> [T]
  func build() async -> DatasetData
  func build()
}

public extension Dataset {
  var dataPublisher:  AnyPublisher<DatasetData, Never> {
    dataPassthroughSubject.eraseToAnyPublisher()
  }
  
  func read<T: FloatingPoint>(path: String, offset: Int, scaleBy: T) -> [T] {
    let url = URL(fileURLWithPath: path)
    
    do {
      let data = try Data(contentsOf: url)
      let array = data.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [UInt8] in
        return Array<UInt8>(pointer[offset..<pointer.count])
      }
      
      let result = array.map { T($0) / scaleBy }

      return result
        
    } catch {
      print("error")
      return []
    }
  }
}
