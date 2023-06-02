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
  
  func test_String_Vectorize() {
    let vectorizer = Vectorizer<String>()
    
    let sentence = "The wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    let vector = vectorizer.vectorize(words)
    
    XCTAssertEqual([1, 2, 3, 4, 5, 1, 6, 7], vector)
  }
  
  func test_String_Vectorize_More_words() {
    let vectorizer = Vectorizer<String>()
    
    let sentence = "The wide road shimmered in the hot sun"
    let words = sentence.components(separatedBy: " ")
    
    let vector = vectorizer.vectorize(words)
    
    XCTAssertEqual([1, 2, 3, 4, 5, 1, 6, 7], vector)
    
    let secondSentence = "The big cat is hot"
    let secondWords = secondSentence.components(separatedBy: " ")

    let secondVector = vectorizer.vectorize(secondWords)

    XCTAssertEqual([1, 8, 9, 10, 6], secondVector)
  }
}
