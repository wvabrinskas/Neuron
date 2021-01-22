import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
  return [
    testCase(NeuronBaseTests.allTests),
    testCase(NeuronClassificationTests.allTests),
    testCase(NeuronPretrainedClassificationTests.allTests),
  ]
}
#endif
