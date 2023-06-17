//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/16/23.
//

import Foundation
import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros
import SwiftDiagnostics


enum NeuronDiagnostic: String, DiagnosticMessage {
  case notAClassForLayer
  case sizeArrayInvalid
  
  var severity: DiagnosticSeverity { return .error }
  
  var message: String {
    switch self {
    case .notAClassForLayer:
      return "'@Layer' can only be applied to a 'class'"
    case .sizeArrayInvalid:
      return "Size array must be of size 3"
    }
  }
  
  var diagnosticID: MessageID {
    MessageID(domain: "MacrosDef", id: rawValue)
  }
}
