//
//  Exportable.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/3/26.
//


import Foundation

/// A type that can be exported to a `.stkns` file on disk.
public protocol Exportable: Codable {
  /// Exports the trainable as a `.stkns` file.
  ///
  /// - Parameters:
  ///   - name: Optional filename prefix.
  ///   - overrite: When `false`, appends a timestamp to avoid overwrite.
  ///   - compress: When `true`, emits compact JSON.
  /// - Returns: URL to the exported model file, or `nil` on write failure.
  @discardableResult
  func export(name: String?, overrite: Bool, compress: Bool) -> URL?
}

/// A type that can be reconstructed from a previously exported file or its raw bytes.
public protocol Importable: Codable {
  
  /// Reconstructs an instance directly from encoded bytes.
  ///
  /// - Parameter data: Serialized bytes produced by `Exportable.export(name:overrite:compress:)`.
  /// - Returns: The decoded instance.
  static func `import`(_ data: Data) -> Self
  
  /// Reconstructs an instance from a previously exported file.
  ///
  /// - Parameter url: File URL pointing to a previously exported file.
  /// - Returns: The decoded instance.
  static func `import`(_ url: URL) -> Self
}
