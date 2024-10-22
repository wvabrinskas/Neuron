//
//  Importer.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/11/24.
//

import ZIPFoundation
import Foundation

protocol ImporterPayload: Sendable {}

protocol Importer {
  associatedtype Payload: ImporterPayload
  func fetch(payload: Payload, precompile: Bool) async throws -> Sequential
}


class BaseImporter<Payload: ImporterPayload>: Importer {
  enum ImporterError: Error {
    case invalidURL
    case invalidResponse
    case invalidData
  }
  
  let session: URLSession
  
  init(session: URLSession = .shared) {
    self.session = session
  }
  
  func fetch(payload: Payload, precompile: Bool = false) async throws -> Sequential {
    // override
    .init()
  }
  
  func download(request: URLRequest) async throws -> Data? {
    let downloaded = try await session.data(for: request)
    
    let response = downloaded.1
    print(response)
    
    let data = downloaded.0
    
    return data
  }
  
  func buildModel(data: Data) throws -> Sequential {
    let network: Result<Sequential, Error> = ExportHelper.buildModel(data)
    
    switch network {
    case .success(let model):
      return model
    case .failure(let error):
      throw error
    }
  }
}
