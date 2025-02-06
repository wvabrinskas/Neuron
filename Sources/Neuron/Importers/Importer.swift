//
//  Importer.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/11/24.
//

import Foundation
import Logger

protocol ImporterPayload: Sendable {}
protocol ResultPayload {
  var model: Sequential { get }
}

protocol Importer {
  associatedtype Payload: ImporterPayload
  associatedtype ResultingPayload: ResultPayload
  func fetch(payload: Payload, precompile: Bool) async throws -> ResultingPayload
}

struct BaseResultPayload: ResultPayload {
  let model: Sequential
}

class BaseImporter<Payload: ImporterPayload, ResultingPayload: ResultPayload>: Importer, Logger {
  var logLevel: LogLevel = .low
  
  enum ImporterError: Error {
    case invalidURL
    case invalidResponse
    case invalidData
  }
  
  let session: URLSession
  
  init(session: URLSession = .shared) {
    self.session = session
  }
  
  func fetch(payload: Payload, precompile: Bool = false) async throws -> ResultingPayload {
    // override
    fatalError("Do not use the BaseImporter, override this")
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
