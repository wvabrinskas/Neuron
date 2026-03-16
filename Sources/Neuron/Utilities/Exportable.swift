//
//  Exportable.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/3/26.
//


import Foundation

/// A type that can be exported to a `.stkns` file on disk.
public protocol Exportable: Codable {
  @discardableResult
  /// Exports the trainable as a `.stkns` file.
  ///
  /// - Parameters:
  ///   - name: Optional filename prefix.
  ///   - overrite: When `false`, appends a timestamp to avoid overwrite.
  ///   - compress: When `true`, emits compact JSON.
  /// - Returns: URL to the exported model file, or `nil` on write failure.
  func export(name: String?, overrite: Bool, compress: Bool) -> URL?
}
