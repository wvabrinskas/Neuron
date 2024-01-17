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
    let vectorizer = Vectorizer<String>()
    
    let sentence = "The wide road shimmered in the hot sun"
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
    
    XCTAssertEqual(sentence.lowercased(), unvectorized)
  }
  
  func test_one_Hot_words() {
    let vectorizer = Vectorizer<String>()
    
    let sentence = "The wide road shimmered in the hot sun"
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
  }
  
  func test_String_Vectorize_Start() {
    let vectorizer = Vectorizer<String>()
    
    let sentence = "The wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    let vector = vectorizer.vectorize(words, format: .start)
    
    XCTAssertEqual([0, 2, 3, 4, 5, 6, 2, 7, 8], vector)
  }
  
  func test_String_Vectorize_End() {
    let vectorizer = Vectorizer<String>()
    
    let sentence = "The wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    let vector = vectorizer.vectorize(words, format: .end)
    
    XCTAssertEqual([2, 3, 4, 5, 6, 2, 7, 8, 1], vector)
  }
  
  func test_String_Vectorize() {
    let vectorizer = Vectorizer<String>()
    
    let sentence = "The wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    let vector = vectorizer.vectorize(words)
    
    XCTAssertEqual([2, 3, 4, 5, 6, 2, 7, 8], vector)
  }
  
  func test_String_Vectorize_More_words() {
    let vectorizer = Vectorizer<String>()
    
    let sentence = "The wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    let vector = vectorizer.vectorize(words)
    
    XCTAssertEqual([2, 3, 4, 5, 6, 2, 7, 8], vector)
    
    let secondSentence = "The big cat is hot"
    let secondWords = secondSentence.components(separatedBy: " ")

    let secondVector = vectorizer.vectorize(secondWords)

    XCTAssertEqual([2, 9, 10, 11, 7], secondVector)
  }
  
  func test_unvectorize() {
    let vectorizer = Vectorizer<String>()
    
    let sentence = "The wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    let vector = vectorizer.vectorize(words)
    
    XCTAssertEqual([2, 3, 4, 5, 6, 2, 7, 8], vector)
    
    let unvector = vectorizer.unvectorize(vector)
    
    let unvectoredSentence = unvector.joined(separator: " ")
    
    XCTAssertEqual(unvectoredSentence, sentence.lowercased())
  }
}
