//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/1/22.
//

import Foundation

internal protocol ModelBuilder {
  func build<TModel: Decodable>(_ json: [AnyHashable : Any]?) -> Result<TModel?, Error>
  func getJSON<T>(_ file: URL) -> Result<T?, Error>
  static func build<TModel: Decodable>(_ json: [AnyHashable : Any]?) -> Result<TModel?, Error>
  static func getJSON<T>(_ file: URL) -> Result<T?, Error>
}

internal enum BuildError: Error {
  case emptyJson
  case pathError
  case error(String)
  
  var localizedDescription: String {
    switch self {
    case .emptyJson:
      return "JSON data was empty"
    case .pathError:
      return "Path could not be found"
    case .error(let message):
      return message
    }
  }
}

internal extension ModelBuilder {
  static func build<TViewModel: Decodable>(_ data: Data?) -> Result<TViewModel?, Error> {
    guard let jsonData = data else {
      return .failure(BuildError.emptyJson)
    }
  
    do {
      let model = try JSONDecoder().decode(TViewModel.self, from: jsonData)
      return .success(model)
    } catch let DecodingError.keyNotFound(key, context) {
      let string = """
      Key '\(key.stringValue)' not found: \(context.debugDescription)
      Coding path: \(context.codingPath.map { $0.stringValue })
      """
      
      return .failure(BuildError.error(string))
    } catch let DecodingError.valueNotFound(type, context) {
      let string = """
      "Value of type '\(type)' not found: \(context.debugDescription)"
      Coding path: \(context.codingPath.map { $0.stringValue })
      """
      
      return .failure(BuildError.error(string))
    } catch let DecodingError.typeMismatch(type, context) {
      let string = """
      "Type mismatch for type '\(type)': \(context.debugDescription)"
      Coding path: \(context.codingPath.map { $0.stringValue })
      """
      
      return .failure(BuildError.error(string))
    } catch let DecodingError.dataCorrupted(context) {
      let string = """
      "Data corrupted: \(context.debugDescription)"
       Coding path: \(context.codingPath.map { $0.stringValue })
      """
      
      return .failure(BuildError.error(string))
    } catch {
      return .failure(BuildError.error(error.localizedDescription))
    }
    
  }
  
  static func build<TViewModel: Decodable>(_ json: [AnyHashable : Any]?) -> Result<TViewModel?, Error> {
    guard let jsonData = json else {
      return .failure(BuildError.emptyJson)
    }
    
    do {
      let modelData = try JSONSerialization.data(withJSONObject: jsonData, options: .prettyPrinted)
      let model = try JSONDecoder().decode(TViewModel.self, from: modelData)
      return .success(model)
    } catch {
      print(error.localizedDescription)
      return .failure(error)
    }
  }
  
  static func getJSON<T>(_ url: URL) -> Result<T?, Error> {
    do {
      let data = try Data(contentsOf: url, options: .mappedIfSafe)
      let json = try? JSONSerialization.jsonObject(with: data, options: .allowFragments)
      return .success(json as? T)
    } catch {
      print("error")
      return .failure(error)
    }
  }
  
  func build<TViewModel: Decodable>(_ json: [AnyHashable : Any]?) -> Result<TViewModel?, Error> {
    guard let jsonData = json else {
      return .failure(BuildError.emptyJson)
    }
    
    do {
      let modelData = try JSONSerialization.data(withJSONObject: jsonData, options: .prettyPrinted)
      let model = try JSONDecoder().decode(TViewModel.self, from: modelData)
      return .success(model)
    } catch {
      print(error.localizedDescription)
      return .failure(error)
    }
  }
  
  func getJSON<T>(_ url: URL) -> Result<T?, Error> {
    do {
      let data = try Data(contentsOf: url, options: .mappedIfSafe)
      let json = try? JSONSerialization.jsonObject(with: data, options: .allowFragments)
      return .success(json as? T)
    } catch {
      print("error")
      return .failure(error)
    }
  }
}

