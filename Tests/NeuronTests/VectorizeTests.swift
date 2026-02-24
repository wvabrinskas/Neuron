//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/1/23.
//

import Foundation
import XCTest
import NumSwift
@testable import Neuron


final class VectorTests: XCTestCase {
  
  func test_on_hot_letters_unvectorize() {
    let names = ["anna",
                 "emma",
                 "elizabeth",
                 "minnie",
                 "margaret",
                 "ida",
                 "alice",
                 "bertha",
                 "sarah"]
    
    let vectorizer = Vectorizer<String>()

    names.forEach { name in
      vectorizer.vectorize(name.fill(with: ".", max: 10).characters)
    }
    
    let testName = "anna"
    let oneHot = vectorizer.oneHot(testName.characters)
    
    let expected = Tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                           [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                           [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                           [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
    
    XCTAssertTrue(oneHot.isValueEqual(to: expected))
    
    let unvectorized = vectorizer.unvectorizeOneHot(oneHot).joined()
    
    XCTAssertEqual(testName, unvectorized)
  }
  
  
  func test_one_hot_letters() {
    let names = ["anna",
                 "emma",
                 "elizabeth",
                 "minnie",
                 "margaret",
                 "ida",
                 "alice",
                 "bertha",
                 "sarah"]
    
    let vectorizer = Vectorizer<String>()

    names.forEach { name in
      vectorizer.vectorize(name.fill(with: ".", max: 10).characters)
    }
    
    let testName = "anna"
    let oneHot = vectorizer.oneHot(testName.characters)
    
    let expected = Tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                           [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                           [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                           [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
    
    XCTAssertTrue(oneHot.isValueEqual(to: expected))
  }
  
  func test_one_Hot_words_unvectorize() {
    let vectorizer = Vectorizer<String>(startAndEndingEncoding: true)
    
    let sentence = "the wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    vectorizer.vectorize(words, format: .start)
    
    let oneHot = vectorizer.oneHot(words)
    
    let expected = Tensor([[[1, 0, 0, 0, 0, 0, 0]],
                           [[0, 1, 0, 0, 0, 0, 0]],
                           [[0, 0, 1, 0, 0, 0, 0]],
                           [[0, 0, 0, 1, 0, 0, 0]],
                           [[0, 0, 0, 0, 1, 0, 0]],
                           [[1, 0, 0, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 0, 1, 0]],
                           [[0, 0, 0, 0, 0, 0, 1]]])
    
    XCTAssertTrue(oneHot.isValueEqual(to: expected))
    
    let unvectorized = vectorizer.unvectorizeOneHot(oneHot).joined(separator: " ")
    
    XCTAssertEqual(sentence, unvectorized)
  }
  
  func test_one_Hot_words() {
    let vectorizer = Vectorizer<String>(startAndEndingEncoding: true)
    
    let sentence = "The wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    vectorizer.vectorize(words, format: .start)
    
    let oneHot = vectorizer.oneHot(words)
    
    let expected = Tensor([[[1, 0, 0, 0, 0, 0, 0, 0]],
                           [[0, 1, 0, 0, 0, 0, 0, 0]],
                           [[0, 0, 1, 0, 0, 0, 0, 0]],
                           [[0, 0, 0, 1, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 1, 0, 0, 0]],
                           [[0, 0, 0, 0, 0, 1, 0, 0]],
                           [[0, 0, 0, 0, 0, 0, 1, 0]],
                           [[0, 0, 0, 0, 0, 0, 0, 1]]])
    
    XCTAssertEqual(expected.storage, oneHot.storage)
  }
  
  func test_String_Vectorize_Start() {
    let vectorizer = Vectorizer<String>(startAndEndingEncoding: true)
    
    let sentence = "the wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    let vector = vectorizer.vectorize(words, format: .start)
    
    XCTAssertEqual([0, 2, 3, 4, 5, 6, 2, 7, 8], vector)
  }
  
  func test_String_Vectorize_End() {
    let vectorizer = Vectorizer<String>(startAndEndingEncoding: true)
    
    let sentence = "the wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    let vector = vectorizer.vectorize(words, format: .end)
    
    XCTAssertEqual([2, 3, 4, 5, 6, 2, 7, 8, 1], vector)
  }
  
  func test_String_Vectorize() {
    let vectorizer = Vectorizer<String>()
    
    let sentence = "the wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    let vector = vectorizer.vectorize(words)
    
    XCTAssertEqual([0, 1, 2, 3, 4, 0, 5, 6], vector)
  }
  
  func test_String_Vectorize_More_words() {
    let vectorizer = Vectorizer<String>()
    
    let sentence = "the wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    let vector = vectorizer.vectorize(words)
    
    XCTAssertEqual([0, 1, 2, 3, 4, 0, 5, 6], vector)
    
    let secondSentence = "the big cat is hot"
    let secondWords = secondSentence.components(separatedBy: " ")

    let secondVector = vectorizer.vectorize(secondWords)

    XCTAssertEqual([0, 7, 8, 9, 5], secondVector)
  }
  
  func test_unvectorize() {
    let vectorizer = Vectorizer<String>()
    
    let sentence = "the wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    let vector = vectorizer.vectorize(words)
    
    XCTAssertEqual([0, 1, 2, 3, 4, 0, 5, 6], vector)
    
    let unvector = vectorizer.unvectorize(vector)
    
    let unvectoredSentence = unvector.joined(separator: " ")
    
    XCTAssertEqual(unvectoredSentence, sentence.lowercased())
  }

  // MARK: - Export / Import

  func test_export_import_roundtrip_url() {
    let vectorizer = Vectorizer<String>()

    let sentence = "the wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    vectorizer.vectorize(words)

    guard let exportURL = vectorizer.export(name: "test-vectorizer-url", overrite: true, compress: true) else {
      XCTFail("Export returned nil URL")
      return
    }

    let imported = Vectorizer<String>.import(exportURL)

    XCTAssertEqual(imported.vector, vectorizer.vector)
    XCTAssertEqual(imported.inverseVector, vectorizer.inverseVector)
    XCTAssertEqual(imported.start, vectorizer.start)
    XCTAssertEqual(imported.end, vectorizer.end)

    // Verify vectorize produces the same indices after import
    let originalVector = vectorizer.vectorize(words)
    let importedVector = imported.vectorize(words)
    XCTAssertEqual(originalVector, importedVector)

    // Verify unvectorize recovers the original words
    let unvectorized = imported.unvectorize(importedVector)
    XCTAssertEqual(unvectorized.joined(separator: " "), sentence)
  }

  func test_export_import_roundtrip_data() {
    let vectorizer = Vectorizer<String>()

    let sentence = "the wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    vectorizer.vectorize(words)

    guard let exportURL = vectorizer.export(name: "test-vectorizer-data", overrite: true, compress: true) else {
      XCTFail("Export returned nil URL")
      return
    }

    guard let data = try? Data(contentsOf: exportURL) else {
      XCTFail("Could not read exported file data")
      return
    }

    let result: Result<Vectorizer<String>, Error> = ExportHelper.buildTokens(data)
    guard case .success(let imported) = result else {
      XCTFail("Import from Data failed")
      return
    }

    XCTAssertEqual(imported.vector, vectorizer.vector)
    XCTAssertEqual(imported.inverseVector, vectorizer.inverseVector)

    let originalVector = vectorizer.vectorize(words)
    let importedVector = imported.vectorize(words)
    XCTAssertEqual(originalVector, importedVector)
  }

  func test_export_import_startAndEndingEncoding_preserved() {
    let vectorizer = Vectorizer<String>(startAndEndingEncoding: true)

    let sentence = "the wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    vectorizer.vectorize(words, format: .start)

    guard let exportURL = vectorizer.export(name: "test-vectorizer-start-end", overrite: true, compress: true) else {
      XCTFail("Export returned nil URL")
      return
    }

    let imported = Vectorizer<String>.import(exportURL)

    XCTAssertEqual(imported.vector, vectorizer.vector)
    XCTAssertEqual(imported.inverseVector, vectorizer.inverseVector)
    XCTAssertEqual(imported.start, 0)
    XCTAssertEqual(imported.end, 1)

    // Verify start/end token formatting works after import
    let withStart = imported.vectorize(words, format: .start)
    XCTAssertEqual(withStart.first, imported.start)

    let withEnd = imported.vectorize(words, format: .end)
    XCTAssertEqual(withEnd.last, imported.end)
  }

  func test_export_import_oneHot_roundtrip() {
    let vectorizer = Vectorizer<String>()

    let names = ["anna", "emma", "elizabeth"]
    names.forEach { name in
      vectorizer.vectorize(name.fill(with: ".", max: 10).characters)
    }

    guard let exportURL = vectorizer.export(name: "test-vectorizer-onehot", overrite: true, compress: true) else {
      XCTFail("Export returned nil URL")
      return
    }

    let imported = Vectorizer<String>.import(exportURL)

    let testName = "anna"
    let originalOneHot = vectorizer.oneHot(testName.characters)
    let importedOneHot = imported.oneHot(testName.characters)

    XCTAssertTrue(originalOneHot.isValueEqual(to: importedOneHot))

    let unvectorized = imported.unvectorizeOneHot(importedOneHot).joined()
    XCTAssertEqual(unvectorized, testName)
  }

  func test_export_import_integer_vectorizer() {
    let vectorizer = Vectorizer<Int>()

    let items = [1, 2, 3, 4, 5, 1, 3]
    vectorizer.vectorize(items)

    guard let exportURL = vectorizer.export(name: "test-vectorizer-int", overrite: true, compress: true) else {
      XCTFail("Export returned nil URL")
      return
    }

    let imported = Vectorizer<Int>.import(exportURL)

    XCTAssertEqual(imported.vector, vectorizer.vector)
    XCTAssertEqual(imported.inverseVector, vectorizer.inverseVector)

    let originalVector = vectorizer.vectorize(items)
    let importedVector = imported.vectorize(items)
    XCTAssertEqual(originalVector, importedVector)

    let unvectorized = imported.unvectorize(importedVector)
    XCTAssertEqual(unvectorized, [1, 2, 3, 4, 5, 1, 3])
  }
}
