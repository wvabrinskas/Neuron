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

/// The protocol that defines how to build a Neuron compatible dataset for training.
/// Example Datasets are:
/// ```
/// MNIST(only num: Int? = nil,
///       label: [Float] = [],
///       zeroCentered: Bool = false)
/// ```
/// ```
/// QuickDrawDataset(objectToGet: QuickDrawObject,
///                    label: [Float],
///                    trainingCount: Int = 1000,
///                    validationCount: Int = 1000,
///                    zeroCentered: Bool = false,
///                    logLevel: LogLevel = .none)
/// ```
public protocol Dataset {
  /// The resulting dataset
  var data: DatasetData { get set }
  /// Indicator that the dataset has loaded
  var complete: Bool { get set }
  /// Combine subject for the dataset
  var dataPassthroughSubject: PassthroughSubject<DatasetData, Never> { get }
  /// Combine publisher for the dataset
  var dataPublisher: AnyPublisher<DatasetData, Never> { get }
  
  /// Read the dataset file from a path
  /// - Parameters:
  ///   - path: Path to the dataset
  ///   - offset: Address offset in bytes to start reading the data from. Many datasets put stuff like size in the first few bytes of the binary dataset file.
  ///   - scaleBy: Value to divide each datapoint by.
  /// - Returns: The dataset as an array of FloatingPoint values
  func read<T: FloatingPoint>(path: String, offset: Int, scaleBy: T) -> [T]
  /// Read the dataset file from a path
  /// - Parameters:
  ///   - data: Data object of the dataset
  ///   - offset: Address offset in bytes to start reading the data from. Many datasets put stuff like size in the first few bytes of the binary dataset file.
  ///   - scaleBy: Value to divide each datapoint by.
  /// - Returns: The dataset as an array of FloatingPoint values
  func read<T: FloatingPoint>(data: Data, offset: Int, scaleBy: T) -> [T]
  
  /// Returns a an array of UInt8 values to project a bitmap
  /// - Parameters:
  ///   - path: Path to the dataset
  ///   - offset: Address offset in bytes to start reading the data from. Many datasets put stuff like size in the first few bytes of the binary dataset file.
  /// - Returns: Array of UInt8 values to project as a bitmap
  func bitmap(path: String, offset: Int) -> [UInt8]
  /// Returns a an array of UInt8 values to project a bitmap
  /// - Parameters:
  ///   - data: Data object of the dataset
  ///   - offset: Address offset in bytes to start reading the data from. Many datasets put stuff like size in the first few bytes of the binary dataset file.
  /// - Returns: Array of UInt8 values to project as a bitmap
  func format(data: Data, offset: Int) -> [UInt8]
  
  /// Async/Await compatable build operation that will read and return the dataset.
  /// - Returns: The `DatasetData` object
  func build() async -> DatasetData
  
  /// Builds the dataset and will publish it to the Combine publisher and downstream subscribers.
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
