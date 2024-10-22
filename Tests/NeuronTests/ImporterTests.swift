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

  func test_remoteImporter() async throws {
    guard isGithubCI == false else {
      XCTAssertTrue(true)
      return
    }

    let urlString = "https://www.kaggle.com/models/williamvabrinskas/pokemon-all-classifier-v2/Other/44m/1/download/pHqaJWS1dWru2r9givMk%2Fversions%2FI5k415ZrwPAfPgEMtNqc%2Ffiles%2Fpokemon-all-classifier_minified_v2.smodel"
    
    let payload = RemotePayload(url: urlString)
    
    let importer = RemoteImporter()
        
    let result = try await importer.fetch(payload: payload, precompile: true)
    
    print(result)

  }
  
}
