#include <metal_stdlib>
using namespace metal;

// perform image convolution using metal shader language
kernel void conv2d(texture2d<float, access::read> inTexture [[ texture(0) ]],
                   texture2d<float, access::write> outTexture [[ texture(1) ]],
                   constant float* filter [[ buffer(0) ]],
                   constant uint2& inSize [[ buffer(1) ]],
                   constant uint2& outSize [[ buffer(2) ]],
                   constant uint2& kernelSize [[ buffer(3) ]],
                   constant uint2& stride [[buffer(4) ]],
                   constant uint2& padding [[ buffer(5) ]],
                   uint2 gid [[ thread_position_in_grid ]]) {
  
  if (gid.x >= outSize.x || gid.y >= outSize.y)
    return;
  
  float sum = 0;
  for (uint i = 0; i < kernelSize.x; i++) {
    for (uint j = 0; j < kernelSize.y; j++) {
      uint x = gid.x * stride.x - padding.x + i;
      uint y = gid.y * stride.y - padding.y + j;
      if (x >= 0 && x < inSize.x && y >= 0 && y < inSize.y) {
        sum += inTexture.read(uint2(x, y)).r * filter[j * kernelSize.y + i];
      }
    }
  }
  
  outTexture.write(float4(sum, sum, sum, 1.0), uint2(gid));
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

