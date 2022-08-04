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
  
  var localizedDescription: String {
    switch self {
    case .emptyJson:
      return "JSON data was empty"
    case .pathError:
      return "Path could not be found"
    }
  }
}

internal extension ModelBuilder {
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

