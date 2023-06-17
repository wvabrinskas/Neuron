import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest
import MacrosImpl

let testMacros: [String: Macro.Type] = [
  "Layerable": LayerMacro.self
]

final class NeuronMacrosTests: XCTestCase {
  
  func testLayerMacroWithSizes() {
    assertMacroExpansion(
            """
            @Layerable(type: .test)
            public class T: Layer {
            }
            """,
            expandedSource:
            """
            
            public class T: Layer {
                public var encodingType: EncodingType = .test
                public var inputSize: TensorSize = TensorSize(array: [28, 28, 0])
                public var outputSize: TensorSize = TensorSize(array: [14, 14, 0])
                public var biasEnabled: Bool = false
                public var trainable: Bool = true
                public var initializer: Initializer?
                public var device: Device = CPU()
                public var weights: Tensor = Tensor()
                public var biases: Tensor = Tensor()
                public init(inputSize: TensorSize = TensorSize(array: []),
                            initializer: InitializerType = .heNormal) {
                  self.inputSize = inputSize
                  self.initializer = initializer.build()
                }
            }
            """,
            macros: testMacros
    )
  }
  
}
