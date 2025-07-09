# Neuron Framework API Documentation

## Table of Contents

1. [Core Classes](#core-classes)
   - [Tensor](#tensor)
   - [TensorContext](#tensorcontext)
   - [TensorSize](#tensorsize)
   - [Sequential](#sequential)
   - [GradientAccumulator](#gradientaccumulator)
2. [Layers](#layers)
   - [Base Layer Classes](#base-layer-classes)
   - [Convolutional Layers](#convolutional-layers)
   - [Dense Layers](#dense-layers)
   - [Activation Layers](#activation-layers)
   - [Normalization Layers](#normalization-layers)
   - [Pooling Layers](#pooling-layers)
   - [Utility Layers](#utility-layers)
   - [Recurrent Layers](#recurrent-layers)
   - [Embedding Layers](#embedding-layers)
3. [Optimizers](#optimizers)
   - [Base Optimizer](#base-optimizer)
   - [Adam](#adam)
   - [SGD](#sgd)
   - [RMSProp](#rmsprop)
   - [Decay Functions](#decay-functions)
   - [Loss Functions](#loss-functions)
4. [Models](#models)
   - [Classifier](#classifier)
   - [GAN](#gan)
   - [WGAN](#wgan)
   - [WGANGP](#wgangp)
   - [RNN](#rnn)
5. [Initialization](#initialization)
   - [InitializerType](#initializertype)
   - [Initializer](#initializer)
6. [Devices](#devices)
   - [CPU](#cpu)
   - [GPU](#gpu)
   - [GPUManager](#gpumanager)
7. [Export/Import](#exportimport)
   - [ExportHelper](#exporthelper)
   - [Importer](#importer)
8. [Utilities](#utilities)
   - [Metrics](#metrics)
   - [Storage](#storage)
   - [Extensions](#extensions)

---

## Core Classes

### Tensor

The fundamental base for all arithmetic in the network. It holds a reference to the backpropagation graph as well as the values of the forward pass.

```swift
public class Tensor: Equatable, Codable
```

#### Properties

- `value: Data` - The actual numerical value of the Tensor as a 3D array `[[[Scalar]]]`
- `shape: [Int]` - Shape of the Tensor as a 1D array `[columns, rows, depth]`
- `isEmpty: Bool` - Returns true if the tensor contains no data
- `label: String` - Description label for the tensor
- `id: UUID` - Generic identifier for the tensor
- `input: Tensor` - Input from the graph
- `context: TensorContext` - Backpropagation context

#### Initializers

```swift
// Default initializer
public init()

// Scalar initializer
public init(_ data: Scalar? = nil, context: TensorContext = TensorContext())

// 1D array initializer
public init(_ data: [Scalar], context: TensorContext = TensorContext())

// 2D array initializer
public init(_ data: [[Scalar]], context: TensorContext = TensorContext())

// 3D array initializer
public init(_ data: Data, context: TensorContext = TensorContext())
```

#### Methods

```swift
// Subscript for tensor slicing
public subscript(_ colRange: some RangeExpression<Int>,
                 _ rowRange: some RangeExpression<Int>,
                 _ depthRange: some RangeExpression<Int>) -> Tensor

// Sets the input graph to this Tensor
public func setGraph(_ tensor: Tensor)

// Calculates gradients in the tensor graph
public func gradients(delta: Tensor) -> Tensor.Gradient

// Detaches tensor from the graph
public func detached() -> Tensor

// Gets scalar value for 1x1x1 tensors
public func asScalar() -> Scalar

// Checks if tensor values are equal
public func isValueEqual(to: Tensor) -> Bool

// Prints the current graph
public func printGraph()
```

#### Nested Types

```swift
// Gradient structure containing input, weight, and bias gradients
public struct Gradient {
    let input: [Tensor]
    let weights: [Tensor]
    let biases: [Tensor]
}
```

### TensorContext

Contains backpropagation information for a given Tensor.

```swift
public struct TensorContext: Codable
```

#### Properties

- `backpropagate: TensorContextFunction` - Function that performs backpropagation

#### Initializers

```swift
public init(backpropagate: TensorContextFunction? = nil)
```

### TensorSize

Represents the dimensions of a tensor.

```swift
public struct TensorSize: Codable
```

#### Properties

- `columns: Int` - Number of columns
- `rows: Int` - Number of rows  
- `depth: Int` - Depth dimension
- `isEmpty: Bool` - Returns true if any dimension is 0

#### Methods

```swift
// Calculate total size
public func totalSize() -> Int

// Get size as array
public func asArray() -> [Int]
```

### Sequential

The main neural network container that organizes layers in sequence.

```swift
public final class Sequential: Trainable
```

#### Properties

- `layers: [Layer]` - Array of layers in the network
- `isCompiled: Bool` - Whether the network has been compiled
- `isTraining: Bool` - Training mode flag
- `device: Device` - Computation device (CPU/GPU)
- `name: String` - Network name

#### Initializers

```swift
// Initialize with variadic layers
public init(_ layers: Layer...)

// Initialize with closure returning layers
public init(_ layers: () -> [Layer])

// Import from URL
public static func import(_ url: URL) -> Self
```

#### Methods

```swift
// Compile the network (calculates layer sizes)
public func compile()

// Make predictions
public func predict(_ data: Tensor, context: NetworkContext) -> Tensor

// Call as function
public func callAsFunction(_ data: Tensor, context: NetworkContext) -> Tensor

// Export weights
public func exportWeights() throws -> [[Tensor]]

// Import weights
public func importWeights(_ weights: [[Tensor]]) throws
```

### GradientAccumulator

Accumulates gradients during training, returning averaged gradients.

```swift
public class GradientAccumulator
```

#### Properties

- `average: Bool` - Flag to enable gradient averaging (default: true)

#### Methods

```swift
// Clear accumulated gradients
public func clear()

// Insert gradient into accumulator
public func insert(_ gradient: Tensor.Gradient)
public func insert(input: Tensor, weights: [Tensor], biases: [Tensor])

// Accumulate and return averaged gradients
public func accumulate(clearAtEnd: Bool = false) -> Tensor.Gradient
```

---

## Layers

### Base Layer Classes

#### Layer Protocol

Base protocol for all layers in the network.

```swift
public protocol Layer: AnyObject, Codable
```

#### Required Properties

- `encodingType: EncodingType` - Type identifier for the layer
- `inputSize: TensorSize` - Input tensor dimensions
- `outputSize: TensorSize` - Output tensor dimensions
- `weights: Tensor` - Layer weights
- `biases: Tensor` - Layer biases
- `biasEnabled: Bool` - Whether biases are enabled
- `trainable: Bool` - Whether layer parameters are trainable
- `isTraining: Bool` - Training mode flag
- `initializer: Initializer?` - Weight initialization method
- `device: Device` - Computation device
- `usesOptimizer: Bool` - Whether to use optimizer for weight updates

#### Required Methods

```swift
// Forward pass
func forward(tensor: Tensor, context: NetworkContext) -> Tensor

// Apply gradients
func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar)

// Export weights
func exportWeights() throws -> [Tensor]

// Import weights
func importWeights(_ weights: [Tensor]) throws
```

#### BaseLayer

Base implementation of the Layer protocol.

```swift
open class BaseLayer: Layer
```

#### BaseConvolutionalLayer

Base class for convolutional layers.

```swift
open class BaseConvolutionalLayer: BaseLayer, ConvolutionalLayer
```

#### Additional Properties

- `filterCount: Int` - Number of filters
- `filters: [Tensor]` - Array of filter tensors
- `filterSize: (rows: Int, columns: Int)` - Filter dimensions
- `strides: (rows: Int, columns: Int)` - Stride values
- `padding: NumSwift.ConvPadding` - Padding type

#### BaseActivationLayer

Base class for activation layers.

```swift
open class BaseActivationLayer: BaseLayer, ActivationLayer
```

### Convolutional Layers

#### Conv2d

2D Convolutional layer for processing 2D data like images.

```swift
public class Conv2d: BaseConvolutionalLayer
```

#### Initializer

```swift
public init(filterCount: Int,
            inputSize: TensorSize? = nil,
            strides: (rows: Int, columns: Int) = (1,1),
            padding: NumSwift.ConvPadding = .valid,
            filterSize: (rows: Int, columns: Int) = (3,3),
            initializer: InitializerType = .heNormal,
            biasEnabled: Bool = false)
```

#### Parameters

- `filterCount` - Number of convolutional filters
- `inputSize` - Input tensor size (required for first layer)
- `strides` - Stride values for convolution
- `padding` - Padding type (.valid, .same)
- `filterSize` - Size of convolutional kernels
- `initializer` - Weight initialization method
- `biasEnabled` - Whether to include bias terms

#### TransConv2d

Transposed 2D Convolutional layer (deconvolution).

```swift
public class TransConv2d: BaseConvolutionalLayer
```

Similar initialization parameters as Conv2d but performs transposed convolution.

### Dense Layers

#### Dense

Fully connected (dense) layer.

```swift
public class Dense: BaseLayer
```

#### Initializer

```swift
public init(_ outputSize: Int,
            inputSize: TensorSize? = nil,
            initializer: InitializerType = .heNormal,
            biasEnabled: Bool = true)
```

#### Parameters

- `outputSize` - Number of output neurons
- `inputSize` - Input tensor size
- `initializer` - Weight initialization method
- `biasEnabled` - Whether to include bias terms

### Activation Layers

#### Activation Types

```swift
public enum Activation: String, Codable {
    case relu, leakyRelu, sigmoid, tanh, softmax, swish, selu, gelu
}
```

#### ReLu

Rectified Linear Unit activation.

```swift
public class ReLu: BaseActivationLayer
```

#### LeakyReLu

Leaky Rectified Linear Unit activation.

```swift
public class LeakyReLu: BaseActivationLayer
```

#### Initializer

```swift
public init(limit: Tensor.Scalar = 0.01, inputSize: TensorSize = TensorSize(array: []))
```

#### Sigmoid

Sigmoid activation function.

```swift
public class Sigmoid: BaseActivationLayer
```

#### Tanh

Hyperbolic tangent activation.

```swift
public class Tanh: BaseActivationLayer
```

#### Softmax

Softmax activation for multi-class classification.

```swift
public class Softmax: BaseActivationLayer
```

#### Swish

Swish activation function.

```swift
public class Swish: BaseActivationLayer
```

#### SeLu

Scaled Exponential Linear Unit activation.

```swift
public class SeLu: BaseActivationLayer
```

#### GeLu

Gaussian Error Linear Unit activation.

```swift
public class GeLu: BaseActivationLayer
```

### Normalization Layers

#### BatchNormalize

Batch normalization layer.

```swift
public class BatchNormalize: BaseLayer
```

#### Initializer

```swift
public init(inputSize: TensorSize = TensorSize(array: []),
            epsilon: Tensor.Scalar = 1e-5,
            momentum: Tensor.Scalar = 0.9)
```

#### Parameters

- `epsilon` - Small constant for numerical stability
- `momentum` - Momentum for running statistics

#### LayerNormalize

Layer normalization.

```swift
public class LayerNormalize: BaseLayer
```

### Pooling Layers

#### MaxPool

Max pooling layer.

```swift
public class MaxPool: BaseLayer
```

#### Initializer

```swift
public init(inputSize: TensorSize = TensorSize(array: []),
            size: (rows: Int, columns: Int) = (2,2),
            strides: (rows: Int, columns: Int) = (2,2))
```

#### AvgPool

Average pooling layer.

```swift
public class AvgPool: BaseLayer
```

Similar initialization as MaxPool but performs average pooling.

### Utility Layers

#### Dropout

Dropout layer for regularization.

```swift
public class Dropout: BaseLayer
```

#### Initializer

```swift
public init(_ rate: Tensor.Scalar, inputSize: TensorSize = TensorSize(array: []))
```

#### Parameters

- `rate` - Dropout rate (0.0 to 1.0)

#### Flatten

Flattens multi-dimensional input to 1D.

```swift
public class Flatten: BaseLayer
```

#### Reshape

Reshapes tensor to specified dimensions.

```swift
public class Reshape: BaseLayer
```

#### Initializer

```swift
public init(_ shape: [Int], inputSize: TensorSize = TensorSize(array: []))
```

### Recurrent Layers

#### LSTM

Long Short-Term Memory layer.

```swift
public class LSTM: BaseLayer
```

#### Initializer

```swift
public init(outputSize: Int,
            inputSize: TensorSize? = nil,
            returnSequences: Bool = false,
            initializer: InitializerType = .heNormal,
            biasEnabled: Bool = true)
```

#### Parameters

- `outputSize` - Number of LSTM units
- `returnSequences` - Whether to return full sequence or last output
- `initializer` - Weight initialization method
- `biasEnabled` - Whether to include bias terms

#### LSTMCell

Individual LSTM cell implementation.

```swift
public class LSTMCell: BaseLayer
```

### Embedding Layers

#### Embedding

Embedding layer for converting indices to dense vectors.

```swift
public class Embedding: BaseLayer
```

#### Initializer

```swift
public init(vocabularySize: Int,
            embeddingSize: Int,
            inputSize: TensorSize? = nil,
            initializer: InitializerType = .heNormal)
```

#### Parameters

- `vocabularySize` - Size of the vocabulary
- `embeddingSize` - Dimension of embedding vectors
- `initializer` - Weight initialization method

---

## Optimizers

### Base Optimizer

Base protocol for all optimizers.

```swift
public protocol Optimizer: AnyObject
```

#### Required Properties

- `trainable: Trainable` - The model being optimized
- `learningRate: Tensor.Scalar` - Learning rate
- `metricsReporter: MetricsReporter?` - Optional metrics reporter
- `decayFunction: DecayFunction?` - Optional learning rate decay

#### Required Methods

```swift
// Forward pass
func forward(_ input: Tensor, context: NetworkContext) -> Tensor

// Backward pass
func backward(_ input: Tensor, expected: Tensor, context: NetworkContext) -> Tensor.Gradient

// Update weights
func step(_ gradients: Tensor.Gradient)

// Zero gradients
func zeroGradients()
```

### Adam

Adam optimizer implementation.

```swift
public class Adam: Optimizer
```

#### Initializer

```swift
public init(_ trainable: Trainable,
            learningRate: Tensor.Scalar = 0.001,
            beta1: Tensor.Scalar = 0.9,
            beta2: Tensor.Scalar = 0.999,
            epsilon: Tensor.Scalar = 1e-8,
            l2Normalize: Bool = false)
```

#### Parameters

- `trainable` - Model to optimize
- `learningRate` - Learning rate
- `beta1` - Exponential decay rate for first moment estimates
- `beta2` - Exponential decay rate for second moment estimates
- `epsilon` - Small constant for numerical stability
- `l2Normalize` - Whether to apply L2 normalization

### SGD

Stochastic Gradient Descent optimizer.

```swift
public class SGD: Optimizer
```

#### Initializer

```swift
public init(_ trainable: Trainable,
            learningRate: Tensor.Scalar = 0.01,
            momentum: Tensor.Scalar = 0.0,
            l2Normalize: Bool = false)
```

#### Parameters

- `momentum` - Momentum factor
- `l2Normalize` - Whether to apply L2 normalization

### RMSProp

RMSProp optimizer implementation.

```swift
public class RMSProp: Optimizer
```

#### Initializer

```swift
public init(_ trainable: Trainable,
            learningRate: Tensor.Scalar = 0.001,
            alpha: Tensor.Scalar = 0.99,
            epsilon: Tensor.Scalar = 1e-8,
            l2Normalize: Bool = false)
```

#### Parameters

- `alpha` - Smoothing constant
- `epsilon` - Small constant for numerical stability
- `l2Normalize` - Whether to apply L2 normalization

### Decay Functions

#### DecayFunction Protocol

```swift
public protocol DecayFunction {
    var decayedLearningRate: Tensor.Scalar { get }
    func reset()
    func step()
}
```

#### ExponentialDecay

Exponential learning rate decay.

```swift
public class ExponentialDecay: DecayFunction
```

#### Initializer

```swift
public init(initialLearningRate: Tensor.Scalar,
            decayRate: Tensor.Scalar,
            decaySteps: Int)
```

### Loss Functions

#### LossFunction

Loss function implementations.

```swift
public enum LossFunction: String, Codable {
    case meanSquaredError, meanAbsoluteError, categoricalCrossentropy, binaryCrossentropy
}
```

#### Methods

```swift
// Calculate loss
public func calculateLoss(predicted: Tensor, expected: Tensor) -> Tensor

// Calculate derivative
public func calculateDerivative(predicted: Tensor, expected: Tensor) -> Tensor
```

---

## Models

### Classifier

High-level classifier model for supervised learning.

```swift
public class Classifier
```

#### Initializer

```swift
public init(optimizer: Optimizer,
            epochs: Int,
            batchSize: Int = 32,
            threadWorkers: Int = 8,
            log: Bool = true)
```

#### Parameters

- `optimizer` - Optimizer to use for training
- `epochs` - Number of training epochs
- `batchSize` - Batch size for training
- `threadWorkers` - Number of worker threads
- `log` - Whether to enable logging

#### Methods

```swift
// Train the model
public func fit(_ trainingData: DatasetType, 
                _ validationData: DatasetType? = nil,
                onEpochCompleted: ((Int) -> Void)? = nil)

// Evaluate model
public func evaluate(_ data: DatasetType) -> [Metric: Tensor.Scalar]
```

### GAN

Generative Adversarial Network implementation.

```swift
public class GAN
```

#### Initializer

```swift
public init(generator: Trainable,
            discriminator: Trainable,
            generatorOptimizer: Optimizer,
            discriminatorOptimizer: Optimizer,
            epochs: Int,
            batchSize: Int = 32,
            threadWorkers: Int = 8)
```

### WGAN

Wasserstein GAN implementation.

```swift
public class WGAN: GAN
```

### WGANGP

Wasserstein GAN with Gradient Penalty.

```swift
public class WGANGP: GAN
```

#### Additional Parameters

- `gradientPenaltyWeight` - Weight for gradient penalty term

### RNN

Recurrent Neural Network model.

```swift
public class RNN
```

#### Initializer

```swift
public init(optimizer: Optimizer,
            epochs: Int,
            batchSize: Int = 32,
            sequenceLength: Int,
            threadWorkers: Int = 8)
```

---

## Initialization

### InitializerType

Weight initialization methods.

```swift
public enum InitializerType: Codable, Equatable {
    case xavierNormal, xavierUniform, heNormal, heUniform, normal(std: Tensor.Scalar)
}
```

#### Methods

```swift
// Create initializer instance
public func build() -> Initializer
```

### Initializer

Weight initializer implementation.

```swift
public struct Initializer
```

#### Methods

```swift
// Calculate single weight value
public func calculate(input: Int, out: Int = 0) -> Tensor.Scalar

// Calculate tensor of weights
public func calculate(size: TensorSize, input: Int, out: Int = 0) -> Tensor
```

---

## Devices

### Device Protocol

```swift
public protocol Device: AnyObject {
    var type: DeviceType { get }
    func activate(_ input: Tensor, _ activation: Activation) -> Tensor
    func derivate(_ input: Tensor, _ activation: Activation) -> Tensor
    // ... other computation methods
}
```

### CPU

CPU computation device.

```swift
public class CPU: Device
```

### GPU

GPU computation device (Metal-based).

```swift
public class GPU: Device
```

### GPUManager

Manages GPU resources and operations.

```swift
public class GPUManager
```

---

## Export/Import

### ExportHelper

Utility for exporting trained models.

```swift
public struct ExportHelper
```

#### Methods

```swift
// Export model to file
public static func export<T: Codable>(filename: String, model: T) -> URL?

// Build model from file
public static func buildModel<T: Codable>(_ url: URL) -> Result<T, Error>
```

### Importer

Model import functionality.

```swift
public struct Importer
```

#### Methods

```swift
// Import model from URL
public static func importModel<T: Codable>(_ url: URL, type: T.Type) -> T?
```

---

## Utilities

### Metrics

Performance metrics for training.

```swift
public enum Metric: String {
    case loss, accuracy, valLoss, valAccuracy, generatorLoss, criticLoss
}
```

#### MetricsReporter

```swift
public class MetricsReporter
```

#### Initializer

```swift
public init(frequency: Int = 1,
            metricsToGather: Set<Metric> = [])
```

#### Properties

- `receive: (([Metric: Tensor.Scalar]) -> Void)?` - Callback for receiving metrics

### Storage

Thread-safe storage utilities.

```swift
public class Storage<T>
```

### Extensions

Various utility extensions for common operations.

#### Extensions+Numbers

```swift
extension Int {
    var asTensorScalar: Tensor.Scalar { get }
}
```

#### Extensions+ThreadSafety

```swift
extension NSLock {
    func with<T>(_ block: () throws -> T) rethrows -> T
}
```

---

## Constants

### Constants

Framework constants and configuration.

```swift
public struct Constants {
    public static var maxWorkers: Int
}
```

---

## Usage Examples

### Basic Neural Network

```swift
import Neuron

// Create network
let network = Sequential {
    [
        Conv2d(filterCount: 32, 
               inputSize: TensorSize(array: [28, 28, 1]),
               padding: .same),
        ReLu(),
        MaxPool(),
        Flatten(),
        Dense(10),
        Softmax()
    ]
}

// Setup optimizer
let optimizer = Adam(network, learningRate: 0.001)

// Create classifier
let classifier = Classifier(optimizer: optimizer,
                          epochs: 10,
                          batchSize: 32)

// Train model
classifier.fit(trainingData, validationData)
```

### Custom Layer Implementation

```swift
class CustomLayer: BaseLayer {
    override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
        // Custom forward pass implementation
        return tensor
    }
    
    override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
        // Custom gradient application
    }
}
```

This documentation provides a comprehensive overview of all classes, functions, and their usage in the Neuron framework. Each section includes detailed parameter descriptions, return values, and usage examples where appropriate.