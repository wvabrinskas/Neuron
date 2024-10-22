//
//  ImporterTests.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/11/24.
//

@testable import Neuron
import Foundation
import XCTest


final class ImporterTests: XCTestCase {

  func test_remoteImporter_not_precompile() async throws {
    guard isGithubCI == false else {
      XCTAssertTrue(true)
      return
    }

    let urlString = "https://williamvabrinskas.com/Neuron/downloads/hamsternames.smodel"
    
    let payload = RemotePayload(url: urlString)
    
    let importer = RemoteImporter()
    importer.logLevel = .high
        
    let result = try await importer.fetch(payload: payload, precompile: false)
        
    XCTAssertFalse(result.model.isCompiled)
  }
  
  func test_remoteImporter_precompile() async throws {
    guard isGithubCI == false else {
      XCTAssertTrue(true)
      return
    }

    let urlString = "https://williamvabrinskas.com/Neuron/downloads/hamsternames.smodel"
    
    let payload = RemotePayload(url: urlString)
    
    let importer = RemoteImporter()
    importer.logLevel = .high
        
    let result = try await importer.fetch(payload: payload, precompile: true)
        
    XCTAssertTrue(result.model.isCompiled)
  }
  
  func test_remoteImporter_cache() async throws {
    guard isGithubCI == false else {
      XCTAssertTrue(true)
      return
    }

    let urlString = "https://williamvabrinskas.com/Neuron/downloads/hamsternames.smodel"
    
    let payload = RemotePayload(url: urlString)
    
    let importer = RemoteImporter()
    importer.logLevel = .high
        
    let result = try await importer.fetch(payload: payload, precompile: true)
        
    XCTAssertTrue(result.model.isCompiled)
    XCTAssertTrue(result.status == .cacheMiss)

    let result2 = try await importer.fetch(payload: payload, precompile: true)
        
    XCTAssertTrue(result2.model.isCompiled)
    XCTAssertTrue(result2.status == .cacheHit)
  }
  
}
