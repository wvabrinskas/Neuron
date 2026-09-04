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

/// A type that can be exported to a `.stkns` file on disk.
public protocol Importable: Codable {
  
  /// Reconstructs a `Sequential` model directly from encoded model data.
  ///
  /// - Parameter data: Serialized model bytes.
  /// - Returns: Decoded `Sequential` instance.
  static func `import`(_ data: Data) -> Self
  
  /// Reconstructs a `Vectorizer` model from a serialized `.stkns` file URL.
  ///
  /// - Parameter url: File URL pointing to a previously exported model.
  /// - Returns: Decoded `Sequential` instance.
  static func `import`(_ url: URL) -> Self
}
