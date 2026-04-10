//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/20/23.
//

import Foundation

public extension String {
  /// Returns the string's Unicode scalars as an array of single-character strings.
  var characters: [String] {
    map { String($0) }
  }

  /// Pads the string to a minimum width by inserting `char` characters.
  ///
  /// - Parameters:
  ///   - char: The fill character. Defaults to a space.
  ///   - max: The minimum character width to achieve.
  ///   - leftAlign: When `true`, appends characters to the right side; when `false`, prepends to the left.
  /// - Returns: A new string padded to at least `max` characters.
  func fill(with char: Character = " ", max: Int, leftAlign: Bool = true) -> String {
    guard max - self.count > 0 else {
      return self
    }
    
    var mapped = self
    for _ in 0..<(max - self.count) {
      let index = leftAlign ? self.endIndex : self.startIndex
      mapped.insert(char, at: index)
    }
    return mapped
  }
}
