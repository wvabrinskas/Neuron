#include <metal_stdlib>
using namespace metal;

uint4 padding_calc(uint2 stride,
                   int padding,
                   uint2 filter_size,
                   uint2 input_size) {
  
  int inputRows = input_size.y;
  int inputColumns = input_size.x;
  
  int strideR = stride.y;
  int strideC = stride.x;
  
  int filterRows = filter_size.y;
  int filterColumns = filter_size.x;
  
  if (padding == 1) {
    float height = (float)inputRows;
    float width = (float)inputColumns;
    
    float outHeight = ceil(height / (float)strideR);
    float outWidth = ceil(width / (float)strideC);
    
    float padAlongHeight = fmax((outHeight - 1) * (float)strideR + (float)filterRows - height, 0);
    float padAlongWidth = fmax((outWidth - 1) * (float)strideC + (float)filterColumns- width, 0);
    
    int paddingT = (int)floor(padAlongHeight / 2);
    int paddingB = (int)padAlongHeight - (float)paddingT;
    int paddingL = (int)floor(padAlongWidth / 2);
    int paddingR = (int)padAlongWidth - (float)paddingL;
    
    // x, y, z, w
    return uint4(paddingT, paddingB, paddingL, paddingR);
    
  } else {
    
    return uint4(0, 0, 0, 0);
  }
}

device float* zero_pad(device float* input,
                       uint2 filter_size,
                       uint2 input_size,
                       uint2 stride) {
  
  uint4 paddingCalc = padding_calc(stride,
                                   1,
                                   filter_size,
                                   input_size);
  
  uint paddingLeft = paddingCalc.z;
  uint paddingRight = paddingCalc.w;
  uint paddingTop = paddingCalc.x;
  uint paddingBottom = paddingCalc.y;
  
  int inputRows = input_size.y;
  int inputColumns = input_size.x;
  
  int padded_row_total = inputRows + paddingLeft + paddingRight;
  int padded_col_total = inputColumns + paddingTop + paddingBottom;
  
  int length = padded_row_total * padded_col_total;
  //float padded[length];
  
  for (int i = 0; i < padded_row_total * padded_col_total; i++) {
   // padded[i] = 0;
  }
  
//  if (padded == NULL || input == NULL)
//    return;
//
  for (int r = 0; r < inputRows; r++) {
    for (int c = 0; c < inputColumns; c++) {
      int padded_c = c + paddingLeft;
      int padded_r = r + paddingTop;
      
      int index = (padded_r  * padded_row_total) + padded_c;
      //padded[index] = input[(r * inputRows) + c];
    }
  }
    
 // memcpy(result, padded, length * sizeof(float));
  
  //free(padded);
}

kernel void conv2d(texture2d<float, access::read> inTexture [[ texture(0) ]],
                   texture2d<float, access::write> outTexture [[ texture(1) ]],
                   constant float* filter [[ buffer(0) ]],
                   constant uint2& inSize [[ buffer(1) ]],
                   constant uint2& outSize [[ buffer(2) ]],
                   constant uint2& kernelSize [[ buffer(3) ]],
                   constant uint2& stride [[buffer(4) ]],
                   constant int& padding [[ buffer(5) ]],
                   uint2 gid [[ thread_position_in_grid ]]) {
  
  uint4 paddingCalc = padding_calc(stride, padding, kernelSize, inSize);
  
  uint paddingLeft = paddingCalc.z;
  uint paddingRight = paddingCalc.w;
  uint paddingTop = paddingCalc.x;
  uint paddingBottom = paddingCalc.y;
  
  uint inputRows = inSize.y;
  uint inputColumns = inSize.x;
  
  uint filterRows = kernelSize.y;
  uint filterColumns = kernelSize.x;
  
  if (gid.x >= outSize.x || gid.y >= outSize.y)
    return;
  
  float sum = 0;
  for (uint i = 0; i < filterColumns; i++) {
    for (uint j = 0; j < filterRows; j++) {
    
      uint x = gid.x * stride.x + i;
      uint y = gid.y * stride.y + j;
      
      if (x >= 0 - paddingLeft && x < inputColumns + paddingRight && y >= 0 - paddingTop && y < inputRows + paddingBottom) {
        sum += inTexture.read(uint2(x, y)).r * filter[j * kernelSize.y + i];
      } else {
        sum = 0;
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

