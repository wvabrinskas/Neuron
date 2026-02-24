//
//  Tokenizing.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/18/26.
//

import Foundation

/// A protocol that defines tokenization capabilities with support for encoding, decoding, and exporting trained models.
public protocol Tokenizing: Codable {
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


public extension Tokenizing {
  @discardableResult
  /// Exports the trainable as a `.stkns` file.
  ///
  /// - Parameters:
  ///   - name: Optional filename prefix.
  ///   - overrite: When `false`, appends a timestamp to avoid overwrite.
  ///   - compress: When `true`, emits compact JSON.
  /// - Returns: URL to the exported model file, or `nil` on write failure.
  func export(name: String?, overrite: Bool, compress: Bool) -> URL? {
    let additional = overrite == false ? "-\(Date().timeIntervalSince1970)" : ""
    
    let filename = (name ?? "tokens") + additional
    
    let dUrl = ExportHelper.getTokens(filename: filename, compress: compress, model: self)
    
    return dUrl
  }
}
