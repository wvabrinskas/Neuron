//
//  Kaggle.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/11/24.
//

import Foundation
import AppleArchive
import Accelerate

struct RemotePayload: ImporterPayload, Sendable {
  let url: String
  
  init(url: String) {
    self.url = url
  }
}


/// Generic importer that downloads a `smodel` file directly from a remove server.
/// Expects the downloaded object to be a `.smodel` file.
final class RemoteImporter: BaseImporter<RemotePayload> {
  override func fetch(payload: RemotePayload, precompile: Bool = false) async throws -> Sequential {
    guard let url = URL(string: payload.url) else {
      throw ImporterError.invalidURL
    }
    
    let urlRequest = URLRequest(url: url)
    
    guard let data = try await download(request: urlRequest) else {
      throw ImporterError.invalidData
    }

    let result = try buildModel(data: data)
    
    if precompile {
      result.compile()
    }
    
    return result
  }
}
