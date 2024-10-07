//
//  Builder.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import Foundation
@_spi(Visualizer) import Neuron


struct BuilderResult {
  var description: String
}

final class Builder {
  func build(_ data: Data) async -> BuilderResult {
    return await withUnsafeContinuation { continuation in
      Task.detached(priority: .userInitiated) {
        let network: Sequential = .import(data)
        network.compile()
        
        continuation.resume(returning: .init(description: network.debugDescription))
      }
    }
  }
}
