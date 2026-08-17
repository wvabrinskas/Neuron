//
//  Importer.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/11/24.
//

import Foundation
import Logger

public protocol ImporterPayload: Sendable {}
public protocol ResultPayload {
  var model: Sequential { get }
}

public protocol Importer {
  associatedtype Payload: ImporterPayload
  associatedtype ResultingPayload: ResultPayload
  func fetch(payload: Payload, precompile: Bool) async throws -> ResultingPayload
}

public struct BaseResultPayload: ResultPayload {
  public let model: Sequential
}

public class BaseImporter<Payload: ImporterPayload, ResultingPayload: ResultPayload>: Importer, Logger {
  public var logLevel: LogLevel = .low
  
  public enum ImporterError: Error {
    case invalidURL
    case invalidResponse
    case invalidData
  }
  
  public let session: URLSession
  
  public init(session: URLSession = .shared) {
    self.session = session
  }
  
  public func fetch(payload: Payload, precompile: Bool = false) async throws -> ResultingPayload {
    // override
    fatalError("Do not use the BaseImporter, override this")
  }

  public func buildModel(data: Data) throws -> Sequential {
    let network: Result<Sequential, Error> = ExportHelper.buildModel(data)
    
    switch network {
    case .success(let model):
      return model
    case .failure(let error):
      throw error
    }
  }
}
