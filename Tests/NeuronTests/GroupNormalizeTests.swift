import XCTest
import NumSwift
@testable import Neuron

/// Isolation test for `GroupNormalize`.
///
/// Case: inputSize (rows: 2, cols: 2, depth: 4), groups = 2  (2 channels per group).
/// Expected values are from `torch.nn.GroupNorm(2, 4, eps: 1e-5, affine: true)`.
///
/// Storage is channel-major: each channel's H×W slice is contiguous, so the
/// nested literals below are [depth=C][rows=H][cols=W].
final class GroupNormalizeTests: XCTestCase {

  private let tol: Tensor.Scalar = 1e-4

  private let inputNested: [[[Tensor.Scalar]]] = [
    [[0.100000, 0.500000], [-0.900000, 1.300000]],   // channel 0
    [[-0.200000, 0.600000], [1.000000, -1.400000]],  // channel 1
    [[0.300000, -0.700000], [1.100000, 1.500000]],   // channel 2
    [[0.400000, 0.800000], [-1.200000, 0.200000]],   // channel 3
  ]
  private let upstreamNested: [[[Tensor.Scalar]]] = [
    [[0.100000, 0.500000], [0.900000, 1.300000]],    // channel 0
    [[0.200000, 0.600000], [1.000000, 1.400000]],    // channel 1
    [[0.300000, 0.700000], [1.100000, 1.500000]],    // channel 2
    [[0.400000, 0.800000], [1.200000, 1.600000]],    // channel 3
  ]
  private let gammaNested: [[[Tensor.Scalar]]] = [[[0.5, 1.0, 1.5, 2.0]]]
  private let betaNested:  [[[Tensor.Scalar]]] = [[[-0.1, 0.0, 0.1, 0.2]]]

  // Expected, channel-major storage order.
  private let expForwardFlat: [Tensor.Scalar] = [-0.114440, 0.116595, -0.692027, 0.578665, -0.375432, 0.548708, 1.010777, -1.761641, 0.100000, -1.692830, 1.534264, 2.251396, 0.439044, 1.395220, -3.385660, -0.039044]
  private let expDInputFlat:  [Tensor.Scalar] = [-0.609574, -0.328825, -0.271787, 0.232671, -0.473582, 0.087914, 0.599697, 0.763487, -1.464145, -0.761953, -0.017928, 0.705180, -1.044324, -0.082171, 0.844124, 1.821217]
  private let expDGamma: [Tensor.Scalar] = [0.912588, -1.201381, 2.366536, -1.816735]
  private let expDBeta:  [Tensor.Scalar] = [2.800000, 3.200000, 3.600000, 4.000000]

  private func makeLayer() -> GroupNormalize {
    let layer = GroupNormalize(groups: 2,
                               inputSize: TensorSize(rows: 2, columns: 2, depth: 4))
    layer.gamma = Tensor(gammaNested)
    layer.beta = Tensor(betaNested)
    return layer
  }

  private func assertClose(_ got: [Tensor.Scalar], _ want: [Tensor.Scalar], _ label: String) {
    XCTAssertEqual(got.count, want.count, "\(label): count mismatch (\(got.count) vs \(want.count))")
    for i in 0..<min(got.count, want.count) {
      XCTAssertEqual(got[i], want[i], accuracy: tol, "\(label)[\(i)]")
    }
  }

  // STEP 1 — forward. Bug here ⇒ normalize() / group statistics.
  func test_forward() {
    let layer = makeLayer()
    let out = layer.forward(tensor: Tensor(inputNested))
    assertClose(out.storage.toArray(), expForwardFlat, "forward")
  }

  // STEP 2 — backward. Bug here (forward passing) ⇒ the γ-fold in backward().
  func test_backward() {
    let layer = makeLayer()
    let input = Tensor(inputNested)
    let out = layer.forward(tensor: input)

    // gradients(delta:) returns Tensor.Gradient with ARRAY members.
    let grads = out.gradients(delta: Tensor(upstreamNested), wrt: input)

    guard let dInput = grads.input.first,
          let dWeight = grads.weights.first else {
      return XCTFail("backward produced no gradients")
    }

    assertClose(dInput.storage.toArray(), expDInputFlat, "dInput")

    // weights packed as beta | gamma along depth (slice 0 = beta, slice 1 = gamma).
    let dBeta = dWeight.depthSliceTensor(0).storage.toArray()
    let dGamma = dWeight.depthSliceTensor(1).storage.toArray()
    assertClose(dBeta, expDBeta, "dBeta")
    assertClose(dGamma, expDGamma, "dGamma")
  }

  // STEP 3 — G == C must equal InstanceNormalize (sanity on the grouping loop).
  func test_reducesToInstanceNorm_whenGroupsEqualChannels() {
    let gn = GroupNormalize(groups: 4, inputSize: TensorSize(rows: 2, columns: 2, depth: 4))
    let inst = InstanceNormalize(inputSize: TensorSize(rows: 2, columns: 2, depth: 4))
    let gOut = gn.forward(tensor: Tensor(inputNested)).storage.toArray()
    let iOut = inst.forward(tensor: Tensor(inputNested)).storage.toArray()
    assertClose(gOut, iOut, "groupNorm(G=C) vs instanceNorm")
  }
}
