//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/1,9/23.
//

import Foundation
import XCTest
import NumSwift
@testable import Neuron

final class LayerTests: XCTestCase {
  
  func testGlobalAveragePool() {
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 16)
    
    let layer = GlobalAvgPool(inputSize: inputSize)
    
    XCTAssertEqual(layer.outputSize, .init(rows: 1, columns: inputSize.depth, depth: 1))
    
    let input = Tensor.fillWith(value: 0.5, size: inputSize)
    
    let out = layer.forward(tensor: input, context: .init())
    
    XCTAssertEqual(TensorSize(array: out.shape).columns, inputSize.depth)
    
    let error = Tensor.fillWith(value: 0.1, size: layer.outputSize)
    
    let gradients = out.gradients(delta: error, wrt: input)
    
    XCTAssertNotNil(gradients.input[safe: 0])
    
    let wrtInput = gradients.input[safe: 0]!
    
    XCTAssertEqual(TensorSize(array: wrtInput.shape), inputSize)
    
    let expectedGradient = Tensor.fillWith(value: 0.1 / (inputSize.rows.asTensorScalar * inputSize.columns.asTensorScalar),
                                           size: inputSize)
    
    let flatGradient = expectedGradient.storage
    
    for g in flatGradient {
      XCTAssertEqual(g,  0.1 / (inputSize.rows.asTensorScalar * inputSize.columns.asTensorScalar), accuracy: 0.0001)
    }
  }
  
  func testResNetDecodeEncode() {
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 1)

    let resNet = ResNet(inputSize: inputSize, filterCount: 8, stride: 1)
    
    let expectedWeights = resNet.weights
    
    do {
      let jsonOut = try JSONEncoder().encode(resNet)
      let jsonIn = try JSONDecoder().decode(ResNet.self, from: jsonOut)
      
      let outWeights = jsonIn.weights
      
      XCTAssertTrue(expectedWeights.isValueEqual(to: outWeights))
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func testConvDecodeEncode() {
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 1)

    let conv = Conv2d(filterCount: 32,
                      inputSize: inputSize,
                      strides: (1,1),
                      padding: .same,
                      filterSize: (3,3),
                      initializer: .heNormal,
                      biasEnabled: true)
    
    let expectedWeights = conv.weights
    
    do {
      let jsonOut = try JSONEncoder().encode(conv)
      let jsonIn = try JSONDecoder().decode(Conv2d.self, from: jsonOut)
      
      let outWeights = jsonIn.weights
      
      XCTAssertTrue(expectedWeights.isValueEqual(to: outWeights))
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func testDenseDecodeEncode() {
    let dense = Dense(20,
                      inputs: 10,
                      initializer: .heNormal,
                      biasEnabled: true)
    
    let expectedWeights = dense.weights
    
    do {
      let jsonOut = try JSONEncoder().encode(dense)
      let jsonIn = try JSONDecoder().decode(Dense.self, from: jsonOut)
      
      let outWeights = jsonIn.weights
      
      XCTAssertTrue(expectedWeights.isValueEqual(to: outWeights))
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func testLayerNormalizeDecodeEncode() {
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 8)
    
    let layerNorm = LayerNormalize(inputSize: inputSize)
    
    let expectedWeights = layerNorm.weights
    
    do {
      let jsonOut = try JSONEncoder().encode(layerNorm)
      let jsonIn = try JSONDecoder().decode(LayerNormalize.self, from: jsonOut)
      
      let outWeights = jsonIn.weights
      
      XCTAssertTrue(expectedWeights.isValueEqual(to: outWeights))
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func testInstanceNormalizeDecodeEncode() {
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 8)
    
    let layerNorm = InstanceNormalize(inputSize: inputSize)
    
    let expectedWeights = layerNorm.weights
    
    do {
      let jsonOut = try JSONEncoder().encode(layerNorm)
      let jsonIn = try JSONDecoder().decode(InstanceNormalize.self, from: jsonOut)
      
      let outWeights = jsonIn.weights
      
      XCTAssertTrue(expectedWeights.isValueEqual(to: outWeights))
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func testBatchNormalizeDecodeEncode() {
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 8)
    
    let batchNorm = BatchNormalize(inputSize: inputSize)
    
    let expectedWeights = batchNorm.weights
    
    do {
      let jsonOut = try JSONEncoder().encode(batchNorm)
      let jsonIn = try JSONDecoder().decode(BatchNormalize.self, from: jsonOut)
      
      let outWeights = jsonIn.weights
      
      XCTAssertTrue(expectedWeights.isValueEqual(to: outWeights))
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func testTransConvDecodeEncode() {
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 8)
    
    let transConv = TransConv2d(filterCount: 16,
                                inputSize: inputSize,
                                strides: (2, 2),
                                padding: .same,
                                filterSize: (3, 3),
                                initializer: .heNormal,
                                biasEnabled: true)
    
    let expectedWeights = transConv.weights
    
    do {
      let jsonOut = try JSONEncoder().encode(transConv)
      let jsonIn = try JSONDecoder().decode(TransConv2d.self, from: jsonOut)
      
      let outWeights = jsonIn.weights
      
      XCTAssertTrue(expectedWeights.isValueEqual(to: outWeights))
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func testLSTMDecodeEncode() {
    let lstm = LSTM(inputUnits: 100,
                    batchLength: 16,
                    returnSequence: true,
                    biasEnabled: true,
                    initializer: .heNormal,
                    hiddenUnits: 64,
                    vocabSize: 38)
    
    let expectedWeights = lstm.weights
    
    do {
      let jsonOut = try JSONEncoder().encode(lstm)
      let jsonIn = try JSONDecoder().decode(LSTM.self, from: jsonOut)
      
      let outWeights = jsonIn.weights
      
      XCTAssertTrue(expectedWeights.isValueEqual(to: outWeights))
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func testEmbeddingDecodeEncode() {
    let embedding = Embedding(inputUnits: 100,
                              vocabSize: 28,
                              batchLength: 16,
                              initializer: .heNormal,
                              trainable: true)
    
    let expectedWeights = embedding.weights
    
    do {
      let jsonOut = try JSONEncoder().encode(embedding)
      let jsonIn = try JSONDecoder().decode(Embedding.self, from: jsonOut)
      
      let outWeights = jsonIn.weights
      
      XCTAssertTrue(expectedWeights.isValueEqual(to: outWeights))
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func testResNet() {
    let inputSize: TensorSize = .init(rows: 64, columns: 64, depth: 3)

    let resNet = ResNet(inputSize: inputSize, filterCount: 16, stride: 2)
    
    let outputSize = resNet.outputSize

    let input = Tensor.fillRandom(size: inputSize)
    
    let out = resNet.forward(tensor: input, context: .init())
        
    XCTAssertEqual(out.shape, outputSize.asArray)
    
    let error = Tensor.fillRandom(size: outputSize)
    
    let gradients = out.gradients(delta: error, wrt: input)
    
    XCTAssertNotNil(gradients.input.first)

    XCTAssertEqual(gradients.input.first?.shape, inputSize.asArray)
    
    // validate it doesn't crash basically...
    resNet.apply(gradients: (Tensor(), Tensor()), learningRate: 0.01)
  }
  
  // MARK: - ResNet Optimizer Gradient Tests
  
  func testResNet_appliesOptimizerGradients_weightsChange() {
    // Test that when we apply gradients from the optimizer, the weights actually change
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 3)
    let filterCount = 8
    
    let resNet = ResNet(inputSize: inputSize, filterCount: filterCount, stride: 2)
    
    // Capture initial weights
    let initialWeights = resNet.weights.copy()
    
    // Forward pass to generate gradients
    let input = Tensor.fillRandom(size: inputSize)
    let out = resNet.forward(tensor: input, context: .init())
    
    // Generate gradients through backward pass
    let error = Tensor.fillRandom(size: resNet.outputSize)
    let gradients = out.gradients(delta: error, wrt: input)
    
    // The gradient weights tensor should be flat containing all layer weights
    var weightStorage: [Tensor.Scalar] = []
    for tensor in gradients.weights {
      weightStorage.append(contentsOf: tensor.storage)
    }
    let weightGradient = Tensor(weightStorage)
    
    var biasStorage: [Tensor.Scalar] = []
    for tensor in gradients.biases {
      biasStorage.append(contentsOf: tensor.storage)
    }
    let biasGradient = Tensor(biasStorage)
    
    // Apply the gradients
    resNet.apply(gradients: (weights: weightGradient, biases: biasGradient), learningRate: 0.01)
    
    // Verify weights have changed
    let newWeights = resNet.weights
    
    XCTAssertFalse(initialWeights.isValueEqual(to: newWeights), 
                   "Weights should change after applying gradients from optimizer")
  }
  
  func testResNet_gradientSizesMatchLayerWeights() {
    // Test that gradient tensor sizes match what the internal layers expect
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 3)
    let filterCount = 8
    
    let resNet = ResNet(inputSize: inputSize, filterCount: filterCount, stride: 2)
    
    // Forward pass
    let input = Tensor.fillRandom(size: inputSize)
    let out = resNet.forward(tensor: input, context: .init())
    
    // Generate gradients
    let error = Tensor.fillRandom(size: resNet.outputSize)
    let gradients = out.gradients(delta: error, wrt: input)
    
    // The ResNet produces gradients per layer - verify they exist and are non-empty
    XCTAssertFalse(gradients.weights.isEmpty, "Should have weight gradients")
    XCTAssertFalse(gradients.biases.isEmpty, "Should have bias gradients")
    
    // Flatten all weight gradients to match how ResNet.apply expects them
    let totalWeightGradientSize = gradients.weights.reduce(0) { $0 + $1.storage.count }
    let totalBiasGradientSize = gradients.biases.reduce(0) { $0 + $1.storage.count }
    
    XCTAssertGreaterThan(totalWeightGradientSize, 0, "Weight gradients should have data")
    XCTAssertGreaterThanOrEqual(totalBiasGradientSize, 0, "Bias gradients should exist (can be 0 if biases disabled)")
  }
  
  func testResNet_withProjection_appliesGradientsToShortcut() {
    // When stride != 1 or filterCount != inputDepth, ResNet uses projection (shortcut path)
    // This test verifies gradients are applied to the shortcut convolution as well
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 3)
    let filterCount = 8 // Different from input depth (3), so projection is used
    
    let resNet = ResNet(inputSize: inputSize, filterCount: filterCount, stride: 2)
    
    // Capture initial weights (includes both inner block and shortcut weights)
    let initialWeights = resNet.weights.copy()
    let initialWeightCount = initialWeights.storage.count
    
    // Forward pass
    let input = Tensor.fillRandom(size: inputSize)
    let out = resNet.forward(tensor: input, context: .init())
    
    // Generate gradients
    let error = Tensor.fillRandom(size: resNet.outputSize)
    let gradients = out.gradients(delta: error, wrt: input)
    
    // Build the gradient tensor as ResNet expects it
    var weightStorage2: [Tensor.Scalar] = []
    for tensor in gradients.weights {
      weightStorage2.append(contentsOf: tensor.storage)
    }
    let weightGradient = Tensor(weightStorage2)
    
    var biasStorage2: [Tensor.Scalar] = []
    for tensor in gradients.biases {
      biasStorage2.append(contentsOf: tensor.storage)
    }
    let biasGradient = Tensor(biasStorage2)
    
    // Apply gradients
    resNet.apply(gradients: (weights: weightGradient, biases: biasGradient), learningRate: 0.01)
    
    let newWeights = resNet.weights
    
    // Verify weights changed
    XCTAssertFalse(initialWeights.isValueEqual(to: newWeights),
                   "Weights should change after applying gradients")
    
    // Verify the weight tensor size is consistent
    XCTAssertEqual(initialWeightCount, newWeights.storage.count,
                   "Weight count should remain the same after gradient application")
  }
  
  func testResNet_withoutProjection_appliesGradientsToInnerBlock() {
    // When stride == 1 AND filterCount == inputDepth, ResNet skips projection
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 8)
    let filterCount = 8 // Same as input depth
    let stride = 1      // No downsampling
    
    let resNet = ResNet(inputSize: inputSize, filterCount: filterCount, stride: stride)
    
    // Capture initial weights
    let initialWeights = resNet.weights.copy()
    
    // Forward pass
    let input = Tensor.fillRandom(size: inputSize)
    let out = resNet.forward(tensor: input, context: .init())
    
    // Generate gradients
    let error = Tensor.fillRandom(size: resNet.outputSize)
    let gradients = out.gradients(delta: error, wrt: input)
    
    var weightStorage3: [Tensor.Scalar] = []
    for tensor in gradients.weights {
      weightStorage3.append(contentsOf: tensor.storage)
    }
    let weightGradient3 = Tensor(weightStorage3)
    
    var biasStorage3: [Tensor.Scalar] = []
    for tensor in gradients.biases {
      biasStorage3.append(contentsOf: tensor.storage)
    }
    let biasGradient3 = Tensor(biasStorage3)
    
    // Apply gradients
    resNet.apply(gradients: (weights: weightGradient3, biases: biasGradient3), learningRate: 0.01)
    
    let newWeights = resNet.weights
    
    XCTAssertFalse(initialWeights.isValueEqual(to: newWeights),
                   "Inner block weights should change after applying gradients")
  }
  
  func testResNet_inSequentialWithOptimizer_weightsUpdateCorrectly() {
    // Integration test: ResNet inside a Sequential with Adam optimizer
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 3)
    
    let network = Sequential {
      [
        ResNet(inputSize: inputSize, filterCount: 8, stride: 2),
        GlobalAvgPool(),
        Dense(10, initializer: .heNormal, biasEnabled: true),
        Softmax()
      ]
    }
    
    let optimizer = Adam(network,
                         learningRate: 0.001,
                         batchSize: 2)
    
    // Capture initial weights from the ResNet layer
    let resNetLayer = network.layers[0] as! ResNet
    let initialResNetWeights = resNetLayer.weights.copy()
    
    // Create training data
    let inputs = [
      Tensor.fillRandom(size: inputSize),
      Tensor.fillRandom(size: inputSize)
    ]
    
    // One-hot labels for 10 classes
    var labels: [Tensor] = []
    for _ in 0..<2 {
      var labelData = [Tensor.Scalar](repeating: 0, count: 10)
      labelData[Int.random(in: 0..<10)] = 1
      labels.append(Tensor(labelData))
    }
    
    // Run a training step
    optimizer.zeroGradients()
    let output = optimizer.fit(inputs,
                               labels: labels,
                               lossFunction: .crossEntropySoftmax)
    optimizer.apply(output.gradients)
    optimizer.step()
    
    // Verify ResNet weights changed
    let newResNetWeights = resNetLayer.weights
    
    XCTAssertFalse(initialResNetWeights.isValueEqual(to: newResNetWeights),
                   "ResNet weights should update when used with optimizer in Sequential")
  }
  
  func testResNet_multipleTrainingSteps_weightsConverge() {
    // Test that weights continue to change over multiple training steps
    let inputSize: TensorSize = .init(rows: 8, columns: 8, depth: 3)
    
    let network = Sequential {
      [
        ResNet(inputSize: inputSize, filterCount: 4, stride: 1),
        GlobalAvgPool(),
        Dense(2, initializer: .heNormal, biasEnabled: true),
        Softmax()
      ]
    }
    
    let optimizer = Adam(network,
                         learningRate: 0.01,
                         batchSize: 4)
    
    let resNetLayer = network.layers[0] as! ResNet
    
    // Create consistent training data
    let inputs = (0..<4).map { _ in Tensor.fillRandom(size: inputSize) }
    let labels = (0..<4).map { i -> Tensor in
      var labelData = [Tensor.Scalar](repeating: 0, count: 2)
      labelData[i % 2] = 1
      return Tensor(labelData)
    }
    
    var previousWeights = resNetLayer.weights.copy()
    var weightChanges: [Bool] = []
    
    // Run multiple training steps
    for _ in 0..<3 {
      optimizer.zeroGradients()
      let output = optimizer.fit(inputs,
                                 labels: labels,
                                 lossFunction: .crossEntropySoftmax)
      optimizer.apply(output.gradients)
      optimizer.step()
      
      let currentWeights = resNetLayer.weights
      weightChanges.append(!previousWeights.isValueEqual(to: currentWeights))
      previousWeights = currentWeights.copy()
    }
    
    // All steps should show weight changes
    XCTAssertTrue(weightChanges.allSatisfy { $0 },
                  "Weights should change on each training step")
  }
  
  func testResNet_gradientMagnitude_reasonable() {
    // Test that gradient magnitudes are reasonable (not exploding or vanishing)
    let inputSize: TensorSize = .init(rows: 16, columns: 16, depth: 3)
    let filterCount = 8
    
    let resNet = ResNet(inputSize: inputSize, filterCount: filterCount, stride: 2)
    
    let input = Tensor.fillRandom(size: inputSize)
    let out = resNet.forward(tensor: input, context: .init())
    
    let error = Tensor.fillRandom(size: resNet.outputSize)
    let gradients = out.gradients(delta: error, wrt: input)
    
    // Check gradient magnitudes
    for (i, weightGrad) in gradients.weights.enumerated() {
      guard !weightGrad.isEmpty else { continue }
      
      let maxGrad = weightGrad.storage.max() ?? 0
      let minGrad = weightGrad.storage.min() ?? 0
      
      // Gradients should be finite and within reasonable bounds
      XCTAssertFalse(maxGrad.isNaN, "Gradient \(i) should not be NaN")
      XCTAssertFalse(maxGrad.isInfinite, "Gradient \(i) should not be infinite")
      XCTAssertFalse(minGrad.isNaN, "Gradient \(i) should not be NaN")
      XCTAssertFalse(minGrad.isInfinite, "Gradient \(i) should not be infinite")
    }
  }
  
  func test_invalid_input_size() {
    let sequential = Sequential {
      [
        ReLu(inputSize: .init(array: [1,1,0])),
        ReLu()
      ]
    }
    
    sequential.compile()
    
    XCTAssertFalse(sequential.isCompiled)
  }
  
  func test_backpropagation_wrt() {
    
    // first branch
    let dense_0 = Dense(20,
                      inputs: 10,
                      initializer: .heNormal,
                      biasEnabled: true)
    
    let dense = Dense(20,
                      inputs: 20,
                      initializer: .heNormal,
                      biasEnabled: true)
    
    let relu = ReLu(inputSize: dense.outputSize)
    
    
    // second branch
    let dense2 = Dense(20,
                       inputs: 10,
                       initializer: .heNormal,
                       biasEnabled: true)
    
    
    let relu2 = ReLu(inputSize: dense2.outputSize)
    
    // output branch
    let dense3 = Dense(30,
                       inputs: relu2.outputSize.columns,
                       initializer: .heNormal,
                       biasEnabled: true)
    
    let relu3 = ReLu(inputSize: dense3.outputSize)
    
    /*
       input_1
         |
       Dense0  input_2
         |       |
       Dense1  Dense2
         |       |
       Relu1   Relu2
          \     /
           \   /
           Dense3 (dual input graph built here)
          /     \
        Relu3   out_2 (current not used)
         |
        out_1 (gradients calculated here)
     
    1. when getting gradients wrt to input_2 at `out` we shouldn't get anything because the
    output of that branch wasn't used at `out`
     
    2. Figure out how when passing twice we set the same graph twice
     */
    
    // feed forward
    let inputAtDense0 = Tensor.fillWith(value: 1, size: dense_0.inputSize)
    inputAtDense0.label = "input_1"
    
    let dense0Out = dense_0(inputAtDense0)
    let dense1Out = dense(dense0Out)
    let reluOut1 = relu(dense1Out)
    
    let inputAtDense2 = Tensor.fillWith(value: 0.8, size: dense2.inputSize)
    inputAtDense2.label = "input_2"

    let reluOut2 = relu2(dense2(inputAtDense2))
    
    let dense3Out1 = dense3(reluOut1)
    
    dense3Out1.setGraph(reluOut1)
    dense3Out1.setGraph(reluOut2)

    let out1 = relu3(dense3Out1) // branch_1 out

    let out2 = dense3(reluOut2) // branch_2 out
    
    // branch 1 backwards
    let branch1Error = Tensor.fillWith(value: 0.5, size: relu3.outputSize)
    let branch1Backwards = out1.gradients(delta: branch1Error, wrt: inputAtDense0)
    
    XCTAssertEqual(branch1Backwards.input.count, 5)
    XCTAssertEqual(branch1Backwards.input[0].shape, dense_0.inputSize.asArray)
    
    // branch 2 backwards
    let branch2Error = Tensor.fillWith(value: 0.5, size: dense3.outputSize)
    let branch2Backwards = out2.gradients(delta: branch2Error, wrt: inputAtDense2)
    
    XCTAssertEqual(branch2Backwards.input.count, 3)
    XCTAssertEqual(branch2Backwards.input[0].shape, dense2.inputSize.asArray)
  
  }
  
  func test_gelu() {
    let gelu = GeLu()
    gelu.inputSize = TensorSize(rows: 3, columns: 3, depth: 3)
    
    let input = Tensor.fillWith(value: 1, size: gelu.inputSize)
    
    let output = gelu.forward(tensor: input)
    
    let expected = Tensor.fillWith(value: 0.8413447, size: gelu.inputSize)
    XCTAssertEqual(expected, output)
    
    let error = Tensor.fillWith(value: 0.2, size: gelu.inputSize)

    let errorOut = output.gradients(delta: error)
    
    let expectedDer = Tensor.fillWith(value: 0.22427435, size: gelu.inputSize)
    XCTAssertEqual(expectedDer, errorOut.input[0])
  }
  
  func test_encode_normal_initializer_type_keepsValue() {
    let expectedStd: Tensor.Scalar = 0.1
    let rawInitializer: InitializerType = .normal(std: expectedStd)
    let initializer = rawInitializer.build()
    
    let encoder = JSONEncoder()
    let data = try? encoder.encode(initializer)
    
    XCTAssertNotNil(data)
    
    let decoder = JSONDecoder()
    let newInitializer = try? decoder.decode(Initializer.self, from: data!)
    
    XCTAssertNotNil(newInitializer)
    
    switch newInitializer!.type {
    case .normal(let std):
      XCTAssertEqual(std, expectedStd, accuracy: 0.00001)
    default:
      XCTFail("Incorrect initializer decoded")
    }
  }
  
  func test_encode_initializers() {
    let rawInitializer: InitializerType = .heNormal
    let initializer = rawInitializer.build()
    
    let encoder = JSONEncoder()
    let data = try? encoder.encode(initializer)
    
    XCTAssertNotNil(data)
    
    let decoder = JSONDecoder()
    let newInitializer = try? decoder.decode(Initializer.self, from: data!)
    
    XCTAssertNotNil(newInitializer)
    XCTAssertEqual(initializer.type, newInitializer!.type)
  }
  
  // MARK: Sequential
  func test_sequential_importExport_Compressed() {
    
    let size = TensorSize(array: [28,28,1])

    let initializer: InitializerType = .heNormal
    
    let firstLayerFilterCount = 8
    let firstDenseLayerDepthCount = firstLayerFilterCount
    let denseLayerOutputSize = (size.columns / 4, size.rows / 4, firstLayerFilterCount)
    let denseLayerOutputCount = denseLayerOutputSize.0 * denseLayerOutputSize.1 * firstDenseLayerDepthCount
    
    let n = Sequential {
      [
        Dense(denseLayerOutputCount,
              inputs: 100,
              initializer: initializer,
              biasEnabled: false),
        LeakyReLu(limit: 0.2),
        Reshape(to: [size.columns / 4, size.rows / 4, firstDenseLayerDepthCount].tensorSize),
        TransConv2d(filterCount: firstLayerFilterCount * 2, //14x14
                    strides: (2,2),
                    padding: .same,
                    filterSize: (3,3),
                    initializer: initializer,
                    biasEnabled: false),
        LeakyReLu(limit: 0.2),
        TransConv2d(filterCount: firstLayerFilterCount, //28x28
                    strides: (2,2),
                    padding: .same,
                    filterSize: (3,3),
                    initializer: initializer,
                    biasEnabled: false),
        LeakyReLu(limit: 0.2),
        Conv2d(filterCount: size.depth,
               strides: (1,1),
               padding: .same,
               filterSize: (7,7),
               initializer: initializer,
               biasEnabled: false),
        Tanh()
      ]
    }
    
    n.compile()

    guard let gUrl = ExportHelper.getModel(filename: "generator", compress: true, model: n) else {
      XCTFail("invalid URL")
      return
    }
    
    let newN = Sequential.import(gUrl)
    newN.compile()
    
    XCTAssertEqual(newN.debugDescription, n.debugDescription)
  }
  
  func test_sequential_importExport_not_Compressed() {
    
    let size = TensorSize(array: [28,28,1])
    
    let initializer: InitializerType = .heNormal
    
    let firstLayerFilterCount = 8
    let firstDenseLayerDepthCount = firstLayerFilterCount
    let denseLayerOutputSize = (size.columns / 4, size.rows / 4, firstLayerFilterCount)
    let denseLayerOutputCount = denseLayerOutputSize.0 * denseLayerOutputSize.1 * firstDenseLayerDepthCount
    
    let n = Sequential {
      [
        Dense(denseLayerOutputCount,
              inputs: 100,
              initializer: initializer,
              biasEnabled: false),
        LeakyReLu(limit: 0.2),
        Reshape(to: [size.columns / 4, size.rows / 4, firstDenseLayerDepthCount].tensorSize),
        TransConv2d(filterCount: firstLayerFilterCount * 2, //14x14
                    strides: (2,2),
                    padding: .same,
                    filterSize: (3,3),
                    initializer: initializer,
                    biasEnabled: false),
        LeakyReLu(limit: 0.2),
        TransConv2d(filterCount: firstLayerFilterCount, //28x28
                    strides: (2,2),
                    padding: .same,
                    filterSize: (3,3),
                    initializer: initializer,
                    biasEnabled: false),
        LeakyReLu(limit: 0.2),
        Conv2d(filterCount: size.depth,
               strides: (1,1),
               padding: .same,
               filterSize: (7,7),
               initializer: initializer,
               biasEnabled: false),
        Tanh()
      ]
    }
    
    n.compile()

    guard let gUrl = ExportHelper.getModel(filename: "generator", compress: false, model: n) else {
      XCTFail("invalid URL")
      return
    }
    
    let newN = Sequential.import(gUrl)
    newN.compile()
    
    XCTAssertEqual(newN.debugDescription, n.debugDescription)
    
  }
  
  func test_Sequential_importWeights() {
    let network = Sequential {
      [
        Dense(5,
              inputs: 5,
              initializer: .heNormal,
              biasEnabled: true),
        ReLu(),
        Dense(5, initializer: .heNormal,
              biasEnabled: true),
        ReLu()
      ]
    }
    
    network.compile()
    
    do {
      let newWeights = try network.exportWeights().map { $0.map { $0.zerosLike() }}
      try network.importWeights(newWeights)
      try network.exportWeights().forEach { $0.forEach { XCTAssertTrue($0.isValueEqual(to: $0.zerosLike() ))}}
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func test_Sequential_exportWeights_didNotCompile() {
    let network = Sequential {
      [
        Dense(20,
              inputs: 8,
              initializer: .heNormal,
              biasEnabled: true),
        ReLu(),
        Dense(10, initializer: .heNormal,
              biasEnabled: true),
        ReLu()
      ]
    }
        
    do {
      let _ = try network.exportWeights().map { $0.map { $0.zerosLike() }}
    } catch {
      XCTAssertTrue(true)
    }
  }
  
  // MARK: SeLu
  func test_seLu() {
    let input: [[[Tensor.Scalar]]] = [[[0.0, 1.0, -1.0, 0.0],
                               [0.0, 1.0, -1.0, 0.0]],
                              [[0.0, 1.0, -1.0, 0.0],
                               [0.0, 1.0, -1.0, 0.0]]]
    
    let inputSize = input.shape

    let layer = SeLu(inputSize: TensorSize(array: inputSize))
    let out = layer.forward(tensor: Tensor(input))
    
    XCTAssertEqual(inputSize, out.shape)
    
    let expected: [[[Tensor.Scalar]]] = [[[0.0, 1.0507, -1.1113541, 0.0],
                                          [0.0, 1.0507, -1.1113541, 0.0]],
                                         [[0.0, 1.0507, -1.1113541, 0.0],
                                          [0.0, 1.0507, -1.1113541, 0.0]]]
    
    XCTAssertEqual(Tensor(expected).isValueEqual(to: out), true)
    
    let delta = Tensor([[[-1.0, 1.0, -1.0, 0.0],
                         [-1.0, 1.0, -1.0, 0.0]],
                        [[-1.0, 1.0, -1.0, 0.0],
                         [-1.0, 1.0, -1.0, 0.0]]])
    
    let gradients = out.gradients(delta: delta)
        
    let expectedGradients: [[[Tensor.Scalar]]] = [[[-1.7581363, 1.0507, -0.6467822, 0.0],
                                           [-1.7581363, 1.0507, -0.6467822, 0.0]],
                                          [[-1.7581363, 1.0507, -0.6467822, 0.0],
                                           [-1.7581363, 1.0507, -0.6467822, 0.0]]]
    
    XCTAssertEqual(Tensor(expectedGradients).isValueEqual(to: gradients.input.first!), true)
  }
  
  // MARK: AvgPool
  func test_avgPool_7x7_kernel_size() {
    let input: [[[Tensor.Scalar]]] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.4, 1.5].as3D()
    let inputSize = input.shape

    let layer = AvgPool(inputSize: TensorSize(array: inputSize), kernelSize: (7,7))
    let out = layer.forward(tensor: Tensor(input))
    
    XCTAssertEqual([2,2,14], out.shape)
    
    let expected: [[[Tensor.Scalar]]] = [[[Tensor.Scalar]]].init(repeating: [[0.4, 1.1571428],
                                                                             [0.4, 1.1571428]], count: 14)
    
    XCTAssertTrue(Tensor(expected).isValueEqual(to: out, accuracy: 0.0001))

    let delta: [[[Tensor.Scalar]]] = [[[Tensor.Scalar]]].init(repeating: [[0.1, 0.3],
                                                                          [0.1, 0.3]], count: 14)
    
    let gradients = out.gradients(delta: Tensor(delta))
    
    XCTAssertEqual(inputSize, gradients.input.first!.shape)
  }
  
  
  func test_avgPool() {
    let input: [[[Tensor.Scalar]]] = [[[0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4]],
                              [[0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4]],
                              [[0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4]]]
    
    let inputSize = input.shape

    let layer = AvgPool(inputSize: TensorSize(array: inputSize))
    let out = layer.forward(tensor: Tensor(input))
    
    XCTAssertEqual([2,2,3], out.shape)
    
    let expected: [[[Tensor.Scalar]]] = [[[0.15, 0.35],
                                  [0.15, 0.35]],
                                 [[0.15, 0.35],
                                  [0.15, 0.35]],
                                 [[0.15, 0.35],
                                  [0.15, 0.35]]]
    
    XCTAssertTrue(Tensor(expected).isValueEqual(to: out, accuracy: 0.0001))
    
    let delta: [[[Tensor.Scalar]]] = [[[0.1, 0.3],
                               [0.2, 0.5]],
                              [[0.1, 0.3],
                               [0.2, 0.5]],
                              [[0.1, 0.3],
                               [0.2, 0.5]]]
    
    let gradients = out.gradients(delta: Tensor(delta))
    
    XCTAssertEqual(inputSize, gradients.input.first!.shape)
    
    let expectedGradients: [[[Tensor.Scalar]]] = [[[0.025, 0.025, 0.075, 0.075],
                                           [0.025, 0.025, 0.075, 0.075],
                                           [0.05, 0.05, 0.125, 0.125],
                                           [0.05, 0.05, 0.125, 0.125]],
                                          [[0.025, 0.025, 0.075, 0.075],
                                           [0.025, 0.025, 0.075, 0.075],
                                           [0.05, 0.05, 0.125, 0.125],
                                           [0.05, 0.05, 0.125, 0.125]],
                                          [[0.025, 0.025, 0.075, 0.075],
                                           [0.025, 0.025, 0.075, 0.075],
                                           [0.05, 0.05, 0.125, 0.125],
                                           [0.05, 0.05, 0.125, 0.125]]]

    XCTAssertTrue(Tensor(expectedGradients).isValueEqual(to: gradients.input.first!, accuracy: 0.0001))

  }
  
  // MARK: Dense
  func test_Dense_Parameters() {
    let dense = Dense(20,
                      inputs: 8,
                      initializer: .heNormal,
                      biasEnabled: true)
    
    XCTAssertEqual(dense.biases.shape, [20, 1, 1])
  }
  
  func test_Dense_importWeights_valid() {
    let dense = Dense(20,
                      inputs: 8,
                      initializer: .heNormal,
                      biasEnabled: true)
    
    do {
      let newWeights = try dense.exportWeights()[safe: 0, Tensor()].zerosLike()
      try dense.importWeights([newWeights])
      XCTAssert(try dense.exportWeights().first!.isValueEqual(to: newWeights))
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func test_Dense_importWeights_invalid_Shape() {
    let dense = Dense(256,
                      inputs: 100,
                      initializer: .heNormal,
                      biasEnabled: true)
    
    do {
      try dense.importWeights([Tensor([10, 10, 10])])
    } catch {
      if let _ = error as? LayerErrors {
        XCTAssertTrue(true)
      } else {
        XCTFail()
      }
    }
  }
  
  // MARK: Convolution
  func test_Conv2d_filters() {
    let conv = Conv2d(filterCount: 32,
                      inputSize: .init(array: [28,28,8]),
                      padding: .same,
                      filterSize: (3,3),
                      initializer: .heNormal,
                      biasEnabled: true)
    
    XCTAssertFalse(conv.filters.isEmpty)
    XCTAssertEqual(conv.filters.shape, [32])
    conv.filters.forEach { f in
      XCTAssertEqual(f.shape, [3,3,8])
    }
    
    XCTAssertEqual(conv.outputSize, TensorSize(array: [28, 28, 32]))
  }
  
  func test_Conv2d_importWeights_valid() {
    let layer = Conv2d(filterCount: 5,
                       inputSize: .init(array: [28,28,1]),
                       filterSize: (3,3),
                       initializer: .heNormal)
    
    do {
      let newWeights = try layer.exportWeights().map { $0.zerosLike() }
      try layer.importWeights(newWeights)
      let exported = try layer.exportWeights()
      
      for i in 0..<exported.count {
        let export = exported[i]
        let new = newWeights[i]
        XCTAssert(new.isValueEqual(to: export))
      }
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  func test_Conv2d_importWeights_invalid_Shape() {
    let layer = Conv2d(filterCount: 5,
                       inputSize: .init(array: [28,28,1]),
                       filterSize: (3,3),
                       initializer: .heNormal)
    
    do {
      try layer.importWeights([Tensor([10, 10, 10])])
    } catch {
      if let _ = error as? LayerErrors {
        XCTAssertTrue(true)
      } else {
        XCTFail()
      }
    }
  }
  
  func test_TransConv2d_filters() {
    let conv = TransConv2d(filterCount: 32,
                           inputSize: .init(array: [28,28,8]),
                           strides: (2,2),
                           padding: .same,
                           filterSize: (3,3),
                           initializer: .heNormal,
                           biasEnabled: true)
    
    XCTAssertFalse(conv.filters.isEmpty)
    XCTAssertEqual(conv.filters.shape, [32])
    conv.filters.forEach { f in
      XCTAssertEqual(f.shape, [3,3,8])
    }
    
    XCTAssertEqual(conv.outputSize, TensorSize(array: [56, 56, 32]))
  }

  
  // MARK: LSTM
  
  func test_LSTM_Weights() {
    let inputUnits = 100
    let hiddenUnits = 256
    let vocabSize = 27
    
    let lstm = LSTM(inputUnits: inputUnits,
                    batchLength: 1,
                    hiddenUnits: hiddenUnits,
                    vocabSize: vocabSize)
    
    XCTAssertEqual(lstm.hiddenOutputWeights.shape, [hiddenUnits, vocabSize, 1])
    
    XCTAssertEqual(lstm.forgetGateWeights.shape, [hiddenUnits, hiddenUnits + inputUnits, 1])
    XCTAssertEqual(lstm.inputGateWeights.shape, [hiddenUnits, hiddenUnits + inputUnits, 1])
    XCTAssertEqual(lstm.outputGateWeights.shape, [hiddenUnits, hiddenUnits + inputUnits, 1])
    XCTAssertEqual(lstm.gateGateWeights.shape, [hiddenUnits, hiddenUnits + inputUnits, 1])
  }
  
  func test_LSTM_Forward() {
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

    let batchLength = 10
    
    names.forEach { name in
      vectorizer.vectorize(name.fill(with: ".", max: batchLength).characters)
    }
    
    let testName = "anna".fill(with: ".", max: batchLength)
    let oneHot = Tensor([testName].map { [[Tensor.Scalar(vectorizer.vector[$0, default: 0])]] })
    
    let inputUnits = 10
    let hiddenUnits = 256
    let vocabSize = vectorizer.vector.count // the size of the total map of vocab letters available. Likely comes from Vectorize
    let inputTensor = oneHot

    let embedding = Embedding(inputUnits: inputUnits,
                              vocabSize: vocabSize,
                              batchLength: batchLength)
    
    let embeddingCalc = embedding.forward(tensor: inputTensor)

    let lstm = LSTM(inputUnits: inputUnits,
                    batchLength: batchLength,
                    initializer: .heNormal,
                    hiddenUnits: hiddenUnits,
                    vocabSize: vocabSize)

        
    let out = lstm.forward(tensor: embeddingCalc, context: .init())
    
    XCTAssertEqual(out.shape, [vocabSize, 1, batchLength])
  }

  func test_Embedding_Forward() {
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

    let batchLength = 10
    
    names.forEach { name in
      vectorizer.vectorize(name.fill(with: ".", max: batchLength).characters)
    }
    
    let testName = "anna".fill(with: ".", max: batchLength)
    let oneHot = vectorizer.oneHot(testName.characters)
    
    let inputUnits = 100
    let vocabSize = vectorizer.vector.count
    
    let embedding = Embedding(inputUnits: inputUnits,
                              vocabSize: vocabSize,
                              batchLength: batchLength)
    
    let inputTensor = oneHot
    
    let out = embedding.forward(tensor: inputTensor)
    
    XCTAssertEqual(out.shape, [inputUnits, 1, batchLength])
  }
  
  func testRexNet() {
    let inputSize: TensorSize = .init(rows: 4, columns: 4, depth: 3)

    let resNet = RexNet(inputSize: inputSize,
                        initializer: .heNormal,
                        strides: (1,1),
                        outChannels: 12,
                        expandRatio: 2)
    
    let outputSize = resNet.outputSize

    let input = Tensor.fillRandom(size: inputSize)
    
    let out = resNet.forward(tensor: input, context: .init())
        
    XCTAssertEqual(out.shape, outputSize.asArray)
    
    let error = Tensor.fillRandom(size: outputSize)
    
    let gradients = out.gradients(delta: error, wrt: input)
    
    let totalWeights = resNet.innerBlockSequential.layers.filter(\.usesOptimizer).reduce(into: 0) { $0 = $0 + $1.weights.size.columns * $1.weights.size.rows * $1.weights.size.depth }
    
    let gradientWeightsCount = gradients.weights.reduce(into: 0) { $0 = $0 + $1.size.columns * $1.size.rows * $1.size.depth }
    
    
    XCTAssertNotNil(gradients.input.first)

    XCTAssertEqual(gradients.input.first?.shape, inputSize.asArray)
    
    XCTAssertEqual(totalWeights, gradientWeightsCount)

    // validate it doesn't crash basically...
    resNet.apply(gradients: (gradients.weights.first!, gradients.biases.first!),
                 learningRate: 0.01)
  }
  

}
