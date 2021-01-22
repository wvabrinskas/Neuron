//
//  ModelBuilder.swift
//  FoxAdKit
//
//  Created by William Vabrinskas on 1/3/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

internal protocol ModelBuilder {
  func build<TModel: Decodable>(_ json: [AnyHashable : Any]?) -> Result<TModel?, Error>
  func getJSON<T>(_ file: String) -> Result<T?, Error>
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
  
  func getJSON<T>(_ file: String) -> Result<T?, Error> {
    do {
      let fileURL = URL(fileURLWithPath: file)
      let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
      let json = try? JSONSerialization.jsonObject(with: data, options: .allowFragments)
      return .success(json as? T)
    } catch {
      print("error")
      return .failure(error)
    }
  }
}

