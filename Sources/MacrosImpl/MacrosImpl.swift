import Foundation
import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros
import SwiftDiagnostics

///// Implementation of the `stringify` macro, which takes an expression
///// of any type and produces a tuple containing the value of that expression
///// and the source code that produced the value. For example
/////
/////     #stringify(x + y)
/////
/////  will expand to
/////
/////     (x + y, "x + y")
//public struct StringifyMacro: ExpressionMacro {
//    public static func expansion(
//        of node: some FreestandingMacroExpansionSyntax,
//        in context: some MacroExpansionContext
//    ) -> ExprSyntax {
//        guard let argument = node.argumentList.first?.expression else {
//            fatalError("compiler bug: the macro does not have any arguments")
//        }
//
//        return "(\(argument), \(literal: argument.description))"
//    }
//}

public struct LayerMacro: MemberMacro {
  public static func expansion(of attribute: AttributeSyntax,
                               providingMembersOf declaration: some DeclGroupSyntax,
                               in context: some MacroExpansionContext) throws -> [DeclSyntax] {
    
    guard let classDecl = declaration.as(ClassDeclSyntax.self) else {
      let structError = Diagnostic(
        node: attribute._syntaxNode,
        message: NeuronDiagnostic.notAClassForLayer
      )
      context.diagnose(structError)
      return []
    }
    
    guard let encodingType = attribute.expression(at: 0)?.trimmed else {
      // TODO: Error
      return []
    }
    
    // these can be force unwrapped since there are default values
    let inputSize = attribute.expression(at: 1) ?? ExprSyntax(literal: [0,0,0])
    let outputSize = attribute.expression(at: 2) ?? ExprSyntax(literal: [0,0,0])
    
    let initializer: DeclSyntax =
    """
    
    public init(inputSize: TensorSize = TensorSize(array: []),
                initializer: InitializerType = .heNormal) {
      self.inputSize = inputSize
      self.initializer = initializer.build()
    }
    """
    return [
      "public var encodingType: EncodingType = \(encodingType)",
      "public var inputSize: TensorSize = TensorSize(array: \(inputSize))",
      "public var outputSize: TensorSize = TensorSize(array: \(outputSize))",
      "public var biasEnabled: Bool = false",
      "public var trainable: Bool = true",
      "public var initializer: Initializer?",
      "public var device: Device = CPU()",
      "public var weights: Tensor = Tensor()",
      "public var biases: Tensor = Tensor()",
      initializer
    ]
  }
  
}

extension AttributeSyntax {
  func expression(at: Int) -> ExprSyntax? {
    var type: ExprSyntax?
    switch argument {
    case .argumentList(let syntax):
      var iterator = syntax.makeIterator()
      var i = 0
      while let next = iterator.next() {
        type = next.expression
        if i == at {
          return type
        }
        i += 1
      }
    default: break
    }
    
    return nil
  }
}

@main
struct NeuronMacrosPlugin: CompilerPlugin {
  let providingMacros: [Macro.Type] = [
    LayerMacro.self
  ]
}
