//
//  ModelBuilder.swift
//  FoxAdKit
//
//  Created by William Vabrinskas on 1/3/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

protocol ModelBuilder {
  func build<TAdModel: Decodable>(_ json: [AnyHashable : Any]?) -> TAdModel?
  func getJSON<T>(_ file: String, _ ext: String) -> T?
}

extension ModelBuilder {
  
  public func build<TViewModel: Decodable>(_ json: [AnyHashable : Any]?) -> TViewModel? {
    guard let jsonData = json else {
      return nil
    }
    
    do {
      let modelData = try JSONSerialization.data(withJSONObject: jsonData, options: .prettyPrinted)
      let model = try JSONDecoder().decode(TViewModel.self, from: modelData)
      return model
    } catch {
      print(error.localizedDescription)
      return nil
    }
  }
  
  public func getJSON<T>(_ file: String, _ ext: String = "json") -> T? {
      do {
        let fileURL = try Resource(name: file, type: ext).url
        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        let json = try? JSONSerialization.jsonObject(with: data, options: .allowFragments)
        return json as? T
      } catch {
        print(error.localizedDescription)
        return nil
      }
    
  }
}

