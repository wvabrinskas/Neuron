//
//  Kaggle.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/11/24.
//

import Foundation
import AppleArchive
import Accelerate

/// The payload for `RemoteImporter`, specifying the remote URL to download the model from.
public struct RemotePayload: ImporterPayload, Sendable {
  /// The URL string pointing to the remote `.smodel` file.
  public let url: String

  /// Creates a new remote payload.
  /// - Parameter url: The URL string pointing to the remote `.smodel` file.
  public init(url: String) {
    self.url = url
  }
}

/// Indicates whether a `RemoteImporter` fetch was served from the local cache or downloaded fresh.
public enum RemoteResultPayloadStatus {
  /// The model data was served from the local in-memory cache.
  case cacheHit
  /// The model data was downloaded from the network.
  case cacheMiss
}

/// The result payload returned by `RemoteImporter`, containing the imported model and its cache status.
public struct RemoteResultPayload: ResultPayload {
  /// The `Sequential` model produced by the import operation.
  public let model: Sequential
  /// Whether the model data was served from cache or freshly downloaded.
  public let status: RemoteResultPayloadStatus
}

/// Generic importer that downloads a `smodel` file directly from a remove server.
/// Expects the downloaded object to be a `.smodel` file.
public final class RemoteImporter: BaseImporter<RemotePayload, RemoteResultPayload> {
  private let cache: NSCache<NSString, NSData> = .init()
  
  /// Downloads (or serves from cache) the `.smodel` file at `payload.url` and builds the model.
  /// - Parameters:
  ///   - payload: The remote URL to fetch the model from.
  ///   - precompile: Whether the resulting model should be compiled immediately after import.
  /// - Returns: The resulting payload containing the built model and cache status.
  /// - Throws: `BaseImporter.ImporterError` if the URL is invalid or the downloaded data is unusable.
  public override func fetch(payload: RemotePayload, precompile: Bool = false) async throws -> RemoteResultPayload {
    guard let url = URL(string: payload.url) else {
      throw ImporterError.invalidURL
    }
    
    let urlRequest = URLRequest(url: url)
    
    let download = try await download(request: urlRequest)
    
    guard let data = download.1 else {
      throw ImporterError.invalidData
    }

    let result = try buildModel(data: data)
    
    if precompile {
      result.compile()
    }
    
    return .init(model: result, status: download.0)
  }
  
  private func download(request: URLRequest, overrideCache: Bool = false) async throws -> (RemoteResultPayloadStatus, Data?) {
    guard let url = request.url else {
      throw ImporterError.invalidURL
    }
    
    if overrideCache == false,
       let cachedObject = cache.object(forKey: url.absoluteString.ns) {
      log(type: .message, priority: .high, message: "\(Self.self) cache hit for \(url.absoluteString)")
      return (.cacheHit, cachedObject.data)
    }
    
    let downloaded = try await session.data(for: request)
      
    let data = downloaded.0
    
    cache.setObject(data.ns, forKey: url.absoluteString.ns)
    
    return (.cacheMiss, data)
  }
  
}

private extension String {
  var ns: NSString {
    NSString(string: self)
  }
}

private extension Data {
  var ns: NSData {
    NSData(data: self)
  }
}

private extension NSData {
  var data: Data {
    Data(referencing: self)
  }
}

