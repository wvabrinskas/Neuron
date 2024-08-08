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

kernel void transConv2d(const device float* input [[ buffer(0) ]],
                        const device float* filter [[ buffer(1) ]],
                        device float* output [[ buffer(2) ]],
                        constant int& inputWidth [[ buffer(3) ]],
                        constant int& inputHeight [[ buffer(4) ]],
                        constant int& inputChannels [[ buffer(5) ]],
                        constant int& kernelSize [[ buffer(6) ]],
                        constant int& outputWidth [[ buffer(7) ]],
                        constant int& outputHeight [[ buffer(8) ]],
                        constant int& outputChannels [[ buffer(9) ]],
                        constant int& strideX [[ buffer(10) ]],
                        constant int& strideY [[ buffer(11) ]],
                        constant int& paddingX [[ buffer(12) ]],
                        constant int& paddingY [[ buffer(13) ]],
                        uint3 gid [[ thread_position_in_grid ]]) {
    
    int x = gid.x;
    int y = gid.y;
    int z = gid.z;
    
    if (x >= outputWidth || y >= outputHeight || z >= outputChannels) {
        return;
    }
    
    float sum = 0.0;
    
    for (int c = 0; c < inputChannels; c++) {
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                int inputX = (x + paddingX - kx) / strideX;
                int inputY = (y + paddingY - ky) / strideY;
                
                if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight &&
                    (x + paddingX - kx) % strideX == 0 && (y + paddingY - ky) % strideY == 0) {
                    int inputIndex = inputY * inputWidth + inputX;
                    int kernelIndex = ((z * inputChannels + c) * kernelSize + ky) * kernelSize + kx;
                    
                    sum += input[c * inputWidth * inputHeight + inputIndex] * filter[kernelIndex];
                }
            }
        }
    }
    
    int outputIndex = (z * outputHeight + y) * outputWidth + x;
    output[outputIndex] = sum;
}

kernel void conv2d(const device float* input [[ buffer(0) ]],
                   const device float* filter [[ buffer(1) ]],
                   device float* output [[ buffer(2) ]],
                   constant int& inputWidth [[ buffer(3) ]],
                   constant int& inputHeight [[ buffer(4) ]],
                   constant int& inputChannels [[ buffer(5) ]],
                   constant int& kernelSize [[ buffer(6) ]],
                   constant int& outputWidth [[ buffer(7) ]],
                   constant int& outputHeight [[ buffer(8) ]],
                   constant int& outputChannels [[ buffer(9) ]],
                   constant int& strideX [[ buffer(10) ]],
                   constant int& strideY [[ buffer(11) ]],
                   constant int& paddingX [[ buffer(12) ]],
                   constant int& paddingY [[ buffer(13) ]],
                   uint3 gid [[ thread_position_in_grid ]]) {
    
    int x = gid.x;
    int y = gid.y;
    int z = gid.z;
    
    if (x >= outputWidth || y >= outputHeight || z >= outputChannels) {
        return;
    }
    
    float sum = 0.0;
    
    for (int c = 0; c < inputChannels; c++) {
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                int inputX = x * strideX + kx - paddingX;
                int inputY = y * strideY + ky - paddingY;
                
                if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight) {
                    int inputIndex = inputY * inputWidth + inputX;
                    int kernelIndex = ((z * inputChannels + c) * kernelSize + ky) * kernelSize + kx;
                    
                    sum += input[c * inputWidth * inputHeight + inputIndex] * filter[kernelIndex];
                }
            }
        }
    }
    
    int outputIndex = (z * outputHeight + y) * outputWidth + x;
    output[outputIndex] = sum;
}

kernel void activation(const device float* data [[ buffer(0) ]],
                       device float* results [[ buffer(1) ]],
                       const device uint& activationType [[ buffer(2) ]],
                       const device float& limit [[ buffer(3) ]],
                       const device uint& width [[ buffer(4) ]],
                       const device uint& height [[ buffer(5) ]],
                       uint2 gid [[ thread_position_in_grid ]]) {
  
  uint x = gid.x;
  uint y = gid.y;
  
  if (x >= width || y >= height) {
    return;
  }
  
  /*
   case .reLu: return 0
   case .leakyRelu: return 1
   case .sigmoid: return 2
   case .swish: return 3
   case .tanh: return 4
   case .softmax: return 5
   case .seLu: return 6
   case .geLu: return 7
   case .none: return 8
   */
  
  uint resultIndex = y * width + x;
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
  } else if (activationType == 6) { //selu
    float alpha = 1.6732632423543772848170429916717;
    float scale = 1.0507009873554804934193349852946;
    if (completeValue > 0) {
      results[resultIndex] = scale * completeValue;
    } else {
      results[resultIndex] = scale * alpha * (exp(completeValue) - 1);
    }
  } else if (activationType == 7) { //gelu
   float cdf = 0.5 * (1.0 + tanh((sqrt(2.0 / M_PI_F) * (completeValue + 0.044715 * pow(completeValue, 3)))));
   results[resultIndex] = completeValue * cdf;
  } else if (activationType == 8 || activationType == 5) { //none & softmax
    results[resultIndex] = completeValue;
  }

}

kernel void derivate(const device float* data [[ buffer(0) ]],
                       device float* results [[ buffer(1) ]],
                       const device uint& activationType [[ buffer(2) ]],
                       const device float& limit [[ buffer(3) ]],
                       const device uint& width [[ buffer(4) ]],
                       const device uint& height [[ buffer(5) ]],
                       uint2 gid [[ thread_position_in_grid ]]) {
  uint x = gid.x;
  uint y = gid.y;
  
  if (x >= width || y >= height) {
    return;
  }
  
  uint resultIndex = y * width + x;
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
    
  } else if (activationType == 6) { //selu
    float alpha = 1.6732632423543772848170429916717;
    float scale = 1.0507009873554804934193349852946;
    if (completeValue > 0) {
      value = scale;
    } else {
      value = scale * alpha * exp(completeValue);
    }
    
  } else if (activationType == 7) { //gelu
    float cdf = 0.5 * (1.0 + tanh((sqrt(2.0 / M_PI_F) * (completeValue + 0.044715 * pow(completeValue, 3)))));
    float pdf = exp(-0.5 * pow(completeValue, 2)) / sqrt(2.0 * M_PI_F);
    value = cdf + completeValue * pdf;
    
  } else if ( else if (activationType == 5 || activationType == 8) { //none & softmax
    value = 1;
  }
  
  results[resultIndex] = value;
}
