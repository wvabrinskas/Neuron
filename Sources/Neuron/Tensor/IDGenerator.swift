//
//  Int64IDGenerator.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/3/26.
//

import Foundation

protocol IDGenerating {
  associatedtype ID: TensorID
  func generate() -> ID
}

class IDGenerator {
  static let shared = IDGenerator()
  private let uuidGenerator: UUIDIDGenerator = .init()
  private let int64Generator: UInt64IDGenerator = .init()
  
  func explicitInt64() -> UInt64 {
    int64Generator.generate()
  }
  
  func explicitUUID() -> UUID {
    uuidGenerator.generate()
  }
  
  func generate<T: TensorID>() -> T {
    if let v = uuidGenerator.generate() as? T {
      return v
    } else if let i = int64Generator.generate() as? T {
      return i
    }
      
    fatalError("ID Type is not supported. Supported types are UUID, and Int64.")
  }
}

final class UUIDIDGenerator: IDGenerating {
  func generate() -> UUID {
    .init()
  }
}

final class UInt64IDGenerator: IDGenerating {
  static let shared = UInt64IDGenerator()
  private var nextID: UInt64 = 0
  
  func generate() -> UInt64 {
    defer { nextID &+= 1 }
    return nextID
  }
}

protocol TensorID: Comparable, Equatable, Codable, Hashable {
  associatedtype ID: Comparable, Equatable, Codable, Hashable
  static func defaultValue() -> ID
}

extension UUID: TensorID {
  static func defaultValue() -> UUID {
    .init()
  }
}

extension UInt64: TensorID {
  static func defaultValue() -> UInt64 {
    0
  }
}
