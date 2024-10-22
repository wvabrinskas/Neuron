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

enum RemoteResultPayloadStatus {
  case cacheHit, cacheMiss
}

struct RemoteResultPayload: ResultPayload {
  let model: Sequential
  let status: RemoteResultPayloadStatus
}

/// Generic importer that downloads a `smodel` file directly from a remove server.
/// Expects the downloaded object to be a `.smodel` file.
final class RemoteImporter: BaseImporter<RemotePayload, RemoteResultPayload> {
  private let cache: NSCache<NSString, CacheObject> = .init()

  private class CacheObject {
    let data: Data
    
    init(data: Data) {
      self.data = data
    }
  }
  
  override func fetch(payload: RemotePayload, precompile: Bool = false) async throws -> RemoteResultPayload {
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
    
    cache.setObject(.init(data: data), forKey: url.absoluteString.ns)
    
    return (.cacheMiss, data)
  }
  
}

private extension String {
  var ns: NSString {
    NSString(string: self)
  }
}

