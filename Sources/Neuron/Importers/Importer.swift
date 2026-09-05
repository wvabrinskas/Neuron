//
//  Importer.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/11/24.
//

import Foundation
import Logger

/// Marker protocol for the input payload passed to an ``Importer``'s `fetch` method.
public protocol ImporterPayload: Sendable {}

/// A payload returned by an ``Importer`` containing the imported model.
public protocol ResultPayload {
  /// The `Sequential` model produced by the import operation.
  var model: Sequential { get }
}

/// Describes a type capable of fetching and building a `Sequential` model from some payload source.
public protocol Importer {
  /// The input payload type used to locate/fetch the model data.
  associatedtype Payload: ImporterPayload
  /// The result payload type returned once the model has been built.
  associatedtype ResultingPayload: ResultPayload

  /// Fetches and builds a model for the given payload.
  /// - Parameters:
  ///   - payload: The input describing where/how to obtain the model data.
  ///   - precompile: Whether the resulting model should be compiled immediately after import.
  /// - Returns: The resulting payload containing the built model.
  /// - Throws: An error if the fetch or model-building step fails.
  func fetch(payload: Payload, precompile: Bool) async throws -> ResultingPayload
}

/// The default `ResultPayload` implementation, wrapping only the imported model.
public struct BaseResultPayload: ResultPayload {
  /// The `Sequential` model produced by the import operation.
  public let model: Sequential
}

/// Base class for importers that fetch model data from a source and build a `Sequential` model from it.
/// Subclass and override `fetch(payload:precompile:)` to implement a concrete import strategy (see `RemoteImporter`).
public class BaseImporter<Payload: ImporterPayload, ResultingPayload: ResultPayload>: Importer, Logger {
  /// The logging verbosity level used when reporting import activity.
  public var logLevel: LogLevel = .low

  /// Errors that can occur while importing a model.
  public enum ImporterError: Error {
    /// The provided URL string could not be parsed into a valid `URL`.
    case invalidURL
    /// The server response was not valid for the request.
    case invalidResponse
    /// The downloaded/provided data could not be used to build a model.
    case invalidData
  }

  /// The `URLSession` used to perform network requests.
  public let session: URLSession

  /// Creates a new importer.
  /// - Parameter session: The `URLSession` used to perform network requests. Defaults to `.shared`.
  public init(session: URLSession = .shared) {
    self.session = session
  }

  /// Fetches and builds a model for the given payload. Must be overridden by subclasses.
  /// - Parameters:
  ///   - payload: The input describing where/how to obtain the model data.
  ///   - precompile: Whether the resulting model should be compiled immediately after import.
  /// - Returns: The resulting payload containing the built model.
  /// - Throws: An error if the fetch or model-building step fails.
  public func fetch(payload: Payload, precompile: Bool = false) async throws -> ResultingPayload {
    // override
    fatalError("Do not use the BaseImporter, override this")
  }

  /// Builds a `Sequential` model from raw `.smodel` data.
  /// - Parameter data: The raw exported model data.
  /// - Returns: The reconstructed `Sequential` model.
  /// - Throws: An error if the data cannot be decoded into a valid model.
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
