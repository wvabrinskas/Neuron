# Dead Code Report – Neuron

Analysis of unused code paths in the Neuron codebase.

---

## Confirmed Dead Code (Never Called)

### 1. **TensorMath.testLarge, testInvalid, testInf, testNaN** (`TensorMath.swift:611-646`)
- **Location**: `Sources/Neuron/Tensor/TensorMath.swift`
- **Description**: Four debug/validation methods on Tensor that check for large values, invalid floats, infinity, and NaN
- **Usage**: Never called from anywhere in the codebase
- **Recommendation**: Remove – debug helpers that were never wired up

### 2. **Tensor.l2Normalized()** (`TensorMath.swift:982-986`)
- **Location**: `Sources/Neuron/Tensor/TensorMath.swift`
- **Description**: Returns L2-normalized tensor
- **Usage**: Never called
- **Recommendation**: Remove or keep if planned for future use (e.g. gradient penalty, normalization layers)

### 3. **Tensor.map(_ transform:)** (`TensorMath.swift:989-995`)
- **Location**: `Sources/Neuron/Tensor/TensorMath.swift`
- **Description**: Element-wise map over tensor scalars
- **Usage**: Never called (`.map` on TensorStorage uses Collection’s map, not this)
- **Recommendation**: Remove – redundant with other patterns

### 4. **TensorStorage.forceCopy()** (`TensorStorage.swift:303-308`)
- **Location**: `Sources/Neuron/Tensor/TensorStorage.swift`
- **Description**: Always copies storage (ignores copy-on-write)
- **Usage**: Never called
- **Recommendation**: Remove – `copy()` is used instead

### 5. **DeviceType.device()** (`Devices.swift:15-22`)
- **Location**: `Sources/Neuron/Devices/Devices.swift`
- **Description**: Converts `DeviceType` enum to `Device` instance (e.g. `.cpu` → `CPU()`)
- **Usage**: Never called
- **Recommendation**: Remove – code uses `device.type` for switching, not this factory

### 6. **Tensor.printGraph(wrt:deep:)** (`Tensor.swift:445-502`)
- **Location**: `Sources/Neuron/Tensor/Tensor.swift`
- **Description**: Debug utility to print computation graph
- **Usage**: Only calls itself recursively; no external caller
- **Recommendation**: Keep as debug utility, or remove if never used manually

---

## Commented-Out / Stub Code

### 7. **GPUManager.conv2d** (commented block, `GPUManager.swift:74-129`)
- **Location**: `Sources/Neuron/Devices/GPUManager.swift`
- **Description**: Large commented-out Metal conv2d implementation
- **Recommendation**: Remove if GPU conv is abandoned, or keep if it’s a reference for future work

---

## Summary

| Item | File | Action |
|------|------|--------|
| testLarge, testInvalid, testInf, testNaN | TensorMath.swift | Remove |
| l2Normalized() | TensorMath.swift | Remove or keep for future |
| map(_ transform:) | TensorMath.swift | Remove |
| forceCopy() | TensorStorage.swift | Remove |
| DeviceType.device() | Devices.swift | Remove |
| printGraph() | Tensor.swift | Optional – debug utility |
| GPUManager commented conv2d | GPUManager.swift | Remove or keep as reference |

---

*Generated from codebase analysis. Re-run verification after changes.*
