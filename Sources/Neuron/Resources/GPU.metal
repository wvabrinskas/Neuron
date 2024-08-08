#include <metal_stdlib>
using namespace metal;

kernel void matmul(const device float* A [[ buffer(0) ]],
                     const device float* B [[ buffer(1) ]],
                     device float* C [[ buffer(2) ]],
                     constant int& M [[ buffer(3) ]],
                     constant int& N [[ buffer(4) ]],
                     constant int& K [[ buffer(5) ]],
                     uint2 gid [[ thread_position_in_grid ]]) {
    int row = gid.y;
    int col = gid.x;
    
    if (row < M && col < N) {
        float sum = 0.0;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}


kernel void transConv3d(const device float* input [[ buffer(0) ]],
                        const device float* kernel [[ buffer(1) ]],
                        device float* output [[ buffer(2) ]],
                        constant int& inputWidth [[ buffer(3) ]],
                        constant int& inputHeight [[ buffer(4) ]],
                        constant int& inputDepth [[ buffer(5) ]],
                        constant int& inputChannels [[ buffer(6) ]],
                        constant int& kernelSize [[ buffer(7) ]],
                        constant int& outputWidth [[ buffer(8) ]],
                        constant int& outputHeight [[ buffer(9) ]],
                        constant int& outputDepth [[ buffer(10) ]],
                        constant int& outputChannels [[ buffer(11) ]],
                        constant int& strideX [[ buffer(12) ]],
                        constant int& strideY [[ buffer(13) ]],
                        constant int& strideZ [[ buffer(14) ]],
                        constant int& paddingX [[ buffer(15) ]],
                        constant int& paddingY [[ buffer(16) ]],
                        constant int& paddingZ [[ buffer(17) ]],
                        uint4 gid [[ thread_position_in_grid ]]) {
    
    int x = gid.x;
    int y = gid.y;
    int z = gid.z;
    int w = gid.w;
    
    if (x >= outputWidth || y >= outputHeight || z >= outputDepth || w >= outputChannels) {
        return;
    }
    
    float sum = 0.0;
    
    for (int c = 0; c < inputChannels; c++) {
        for (int kz = 0; kz < kernelSize; kz++) {
            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    int inputX = (x + paddingX - kx) / strideX;
                    int inputY = (y + paddingY - ky) / strideY;
                    int inputZ = (z + paddingZ - kz) / strideZ;
                    
                    if (inputX >= 0 && inputX < inputWidth &&
                        inputY >= 0 && inputY < inputHeight &&
                        inputZ >= 0 && inputZ < inputDepth &&
                        (x + paddingX - kx) % strideX == 0 &&
                        (y + paddingY - ky) % strideY == 0 &&
                        (z + paddingZ - kz) % strideZ == 0) {
                        int inputIndex = ((c * inputDepth + inputZ) * inputHeight + inputY) * inputWidth + inputX;
                        int kernelIndex = (((w * inputChannels + c) * kernelSize + kz) * kernelSize + ky) * kernelSize + kx;
                        
                        sum += input[inputIndex] * kernel[kernelIndex];
                    }
                }
            }
        }
    }
    
    int outputIndex = ((w * outputDepth + z) * outputHeight + y) * outputWidth + x;
    output[outputIndex] = sum;
}


kernel void conv3d(const device float* input [[ buffer(0) ]],
                   const device float* kernel [[ buffer(1) ]],
                   device float* output [[ buffer(2) ]],
                   constant int& inputWidth [[ buffer(3) ]],
                   constant int& inputHeight [[ buffer(4) ]],
                   constant int& inputDepth [[ buffer(5) ]],
                   constant int& inputChannels [[ buffer(6) ]],
                   constant int& kernelSize [[ buffer(7) ]],
                   constant int& outputWidth [[ buffer(8) ]],
                   constant int& outputHeight [[ buffer(9) ]],
                   constant int& outputDepth [[ buffer(10) ]],
                   constant int& outputChannels [[ buffer(11) ]],
                   constant int& strideX [[ buffer(12) ]],
                   constant int& strideY [[ buffer(13) ]],
                   constant int& strideZ [[ buffer(14) ]],
                   constant int& paddingX [[ buffer(15) ]],
                   constant int& paddingY [[ buffer(16) ]],
                   constant int& paddingZ [[ buffer(17) ]],
                   uint4 gid [[ thread_position_in_grid ]]) {
    
    int x = gid.x;
    int y = gid.y;
    int z = gid.z;
    int w = gid.w;
    
    if (x >= outputWidth || y >= outputHeight || z >= outputDepth || w >= outputChannels) {
        return;
    }
    
    float sum = 0.0;
    
    for (int c = 0; c < inputChannels; c++) {
        for (int kz = 0; kz < kernelSize; kz++) {
            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    int inputX = x * strideX + kx - paddingX;
                    int inputY = y * strideY + ky - paddingY;
                    int inputZ = z * strideZ + kz - paddingZ;
                    
                    if (inputX >= 0 && inputX < inputWidth &&
                        inputY >= 0 && inputY < inputHeight &&
                        inputZ >= 0 && inputZ < inputDepth) {
                        int inputIndex = ((c * inputDepth + inputZ) * inputHeight + inputY) * inputWidth + inputX;
                        int kernelIndex = (((w * inputChannels + c) * kernelSize + kz) * kernelSize + ky) * kernelSize + kx;
                        
                        sum += input[inputIndex] * kernel[kernelIndex];
                    }
                }
            }
        }
    }
    
    int outputIndex = ((w * outputDepth + z) * outputHeight + y) * outputWidth + x;
    output[outputIndex] = sum;
}



kernel void activation(const device float* data [[ buffer(0) ]],
                       device float* results [[ buffer(1) ]],
                       const device uint& activationType [[ buffer(2) ]],
                       const device float& limit,
                       const uint tgPos [[ threadgroup_position_in_grid ]],
                       const uint tPerTg [[ threads_per_threadgroup ]],
                       const uint tPos [[ thread_position_in_threadgroup ]]) {
  
  uint resultIndex = tgPos * tPerTg + tPos;
  
  float completeValue = data[resultIndex];

  if (activationType == 0) { //relu
    results[resultIndex] = max((float)0, completeValue);
    
  } else if (activationType == 1) { //leaky relu
    if (completeValue < 0) {
      results[resultIndex] = limit * completeValue;
    } else {
      results[resultIndex] = completeValue;
    }
      
  } else if (activationType == 2) { //sigmoid
    results[resultIndex] = 1.0 / (1.0 + exp(-completeValue));

  } else if (activationType == 3) { //swish
    float sigmoid = 1.0 / (1.0 + exp(-completeValue));
    results[resultIndex] = completeValue * sigmoid;
    
  } else if (activationType == 4) { //tanH
    float denom = 1.0 + exp(-2 * completeValue);
    results[resultIndex] = (2.0 / denom) - 1.0;
    
  } else if (activationType == 5) { //none
    results[resultIndex] = completeValue;
  }

}

kernel void derivate(const device float* data [[ buffer(0) ]],
                     device float* results [[ buffer(1) ]],
                     const device uint& activationType [[ buffer(2) ]],
                     const device float& limit,
                     const uint tgPos [[ threadgroup_position_in_grid ]],
                     const uint tPerTg [[ threads_per_threadgroup ]],
                     const uint tPos [[ thread_position_in_threadgroup ]]) {

  uint resultIndex = tgPos * tPerTg + tPos;
  
  float completeValue = data[resultIndex];
  
  float value = completeValue;
  
  if (activationType == 0) { //relu
    if (completeValue >= 0) {
      value = 1;
    } else {
      value = 0;
    }
    
  } else if (activationType == 1) { //leaky relu
    if (completeValue > 0) {
      value = 1;
    } else {
      value = limit;
    }
    
  } else if (activationType == 2) { //sigmoid
    float sig = 1.0 / (1.0 + exp(-completeValue));
    value = sig * (1 - sig);
    
  } else if (activationType == 3) { //swish
    value = (exp(-completeValue) * (completeValue + 1) + 1) / pow((1 + exp(-completeValue)), 2);
    
  } else if (activationType == 4) { //tanH
    float denom = 1.0 + exp(-2 * completeValue);
    float tanActivate = (2.0 / denom) - 1.0;
    value = 1 - (pow(tanActivate, 2));
    
  } else if (activationType == 5) { //none
    results[resultIndex] = 1;
  }
  
  results[resultIndex] = value;
}

kernel void conv2d(const device float* input [[ buffer(0) ]],
                   const device float* kernel [[ buffer(1) ]],
                   device float* output [[ buffer(2) ]],
                   constant int& inputWidth [[ buffer(3) ]],
                   constant int& inputHeight [[ buffer(4) ]],
                   constant int& inputChannels [[ buffer(5) ]],
                   constant int& kernelSize [[ buffer(6) ]],
                   constant int& outputWidth [[ buffer(7) ]],
                   constant int& outputHeight [[ buffer(8) ]],
                   constant int& outputChannels [[ buffer(9) ]],
                   uint3 gid [[ thread_position_in_grid ]]) {
    
    int x = gid.x;
    int y = gid.y;
    int z = gid.z;
    
    if (x >= outputWidth || y >= outputHeight || z >= outputChannels) {
        return;
    }
    
    float sum = 0.0;
    int halfKernel = kernelSize / 2;
    
    for (int c = 0; c < inputChannels; c++) {
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                int ix = x + kx - halfKernel;
                int iy = y + ky - halfKernel;
                
                if (ix >= 0 && ix < inputWidth && iy >= 0 && iy < inputHeight) {
                    int inputIndex = (c * inputHeight + iy) * inputWidth + ix;
                    int kernelIndex = ((z * inputChannels + c) * kernelSize + ky) * kernelSize + kx;
                    
                    sum += input[inputIndex] * kernel[kernelIndex];
                }
            }
        }
    }
    
    int outputIndex = (z * outputHeight + y) * outputWidth + x;
    output[outputIndex] = sum;
}