//
//  FileManager.swift
//  Nameley
//
//  Created by William Vabrinskas on 12/22/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

public struct ExportManager {
  
  /// Accepts an array of T generics and returns the array as a CSV url that was saved to disk
  /// - Parameter data: The data to encode to a CSV
  /// - Returns: The url that points to the written file. nil if it failed to write
  public static func getCSV<T>(filename: String = "file", _ data: [T]) -> URL? {
    var stringData = data.description
    stringData = stringData.replacingOccurrences(of: ",", with: ",\n")
    
    stringData.remove(at: stringData.startIndex)

    let endIndex = stringData.index(stringData.startIndex, offsetBy: stringData.count - 1)
    stringData.remove(at: endIndex)
    
    let fileManager = FileManager.default
    
    do {
      
      let path = try fileManager.url(for: .documentDirectory, in: .allDomainsMask, appropriateFor: nil , create: false )
      let fileName = "\(filename).csv"
      let fileURL = path.appendingPathComponent(fileName)
      
      try stringData.write(to: fileURL, atomically: true , encoding: .utf8)
      
      return fileURL
      
    } catch {
      print("error creating file")
      return nil
      
    }

  }
  
  public static func getModel<T: Codable>(filename: String = "model", model: T) -> URL? {
    let fileManager = FileManager.default

    do {
      let encoder = JSONEncoder()
      encoder.outputFormatting = .prettyPrinted
      let dict = try encoder.encode(model)
      
      let path = try fileManager.url(for: .documentDirectory, in: .allDomainsMask, appropriateFor: nil, create: false)
      let fileName = "\(filename).smodel"
      
      let fileURL = path.appendingPathComponent(fileName)
      
      try dict.write(to: fileURL)
      
      return fileURL
      
    } catch {
      print("error creating file")
      return nil
      
    }
    return nil
  }
}
