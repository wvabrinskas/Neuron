//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/1/22.
//

import Foundation

public struct ExportHelper: ModelBuilder {
  
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
  
  /// Encodes and writes a Codable vectorizor / tokenizer to a `.stkns` file.
  ///
  /// - Parameters:
  ///   - filename: Output filename without extension.
  ///   - compress: When `true`, emits compact JSON; otherwise pretty-printed JSON.
  ///   - model: Codable model to serialize.
  /// - Returns: File URL for the written model, or `nil` on failure.
  public static func getTokens<T: Codable>(filename: String = "tokens", compress: Bool = true, model: T) -> URL? {
    let fileManager = FileManager.default

    do {
      let encoder = JSONEncoder()
      
      if compress == false {
        encoder.outputFormatting = .prettyPrinted
      }
      
      let dict = try encoder.encode(model)
      
      let path = try fileManager.url(for: .documentDirectory, in: .allDomainsMask, appropriateFor: nil, create: false)
      let fileName = "\(filename).stkns"
      let fileURL = path.appendingPathComponent(fileName)
      
      try dict.write(to: fileURL)

      return fileURL
      
    } catch {
      print("error creating file")
      return nil
      
    }
  }
  
  /// Encodes and writes a Codable model to a `.smodel` file.
  ///
  /// - Parameters:
  ///   - filename: Output filename without extension.
  ///   - compress: When `true`, emits compact JSON; otherwise pretty-printed JSON.
  ///   - model: Codable model to serialize.
  /// - Returns: File URL for the written model, or `nil` on failure.
  public static func getModel<T: Codable>(filename: String = "model", compress: Bool = true, model: T) -> URL? {
    let fileManager = FileManager.default

    do {
      let encoder = JSONEncoder()
      
      if compress == false {
        encoder.outputFormatting = .prettyPrinted
      }
      
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
  }
  
  @_spi(Visualizer)
  /// Builds a trainable model instance from a serialized model file URL.
  ///
  /// - Parameter url: URL to a `.smodel` file.
  /// - Returns: Success with decoded model, or failure with decode/load error.
  public static func buildModel<T: Trainable>(_ url: URL) -> Result<T, Error> {
    do {
      let data = try Data(contentsOf: url, options: .mappedIfSafe)
      return buildModel(data)
    } catch {
      return .failure(error)
    }
  }
  
  internal static func buildModel<T: Trainable>(_ data: Data) -> Result<T, Error> {
    do {
      let modelResult: Result<T?, Error> = self.build(data)
      if let model = try modelResult.get() {
        return .success(model)
      }
      
      return .failure(BuildError.emptyJson)
      
    } catch {
      return .failure(error)
    }
  }
  
  @_spi(Visualizer)
  /// Builds a trainable model instance from a serialized model file URL.
  ///
  /// - Parameter url: URL to a `.smodel` file.
  /// - Returns: Success with decoded model, or failure with decode/load error.
  public static func buildTokens<T: Tokenizing>(_ url: URL) -> Result<T, Error> {
    do {
      let data = try Data(contentsOf: url, options: .mappedIfSafe)
      return buildTokens(data)
    } catch {
      return .failure(error)
    }
  }
  
  internal static func buildTokens<T: Tokenizing>(_ data: Data) -> Result<T, Error> {
    do {
      let modelResult: Result<T?, Error> = self.build(data)
      if let model = try modelResult.get() {
        return .success(model)
      }
      
      return .failure(BuildError.emptyJson)
      
    } catch {
      return .failure(error)
    }
  }
}

