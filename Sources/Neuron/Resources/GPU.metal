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

kernel void derivation(texture2d_array<float, access::read> inTexture [[ texture(0) ]],
                       texture2d_array<float, access::write> outTexture [[ texture(1) ]],
                       const device uint& activationType [[ buffer(2) ]],
                       const device float& limit,
                       uint3 gid [[ thread_position_in_grid ]]) {
  
  uint2 coord = uint2(gid.x, gid.y);
  uint slice = gid.z;
  
  float4 completeValue = inTexture.read(coord, slice);
  
  if (activationType == 0) { //relu
    if (completeValue.x >= 0.0 && completeValue.y >= 0.0 && completeValue.z >= 0.0 && completeValue.w >= 0.0) {
      completeValue = 1;
    } else {
      completeValue = 0;
    }
    
  } else if (activationType == 1) { //leaky relu
    if (completeValue.x > 0.0 && completeValue.y > 0.0 && completeValue.z > 0.0 && completeValue.w > 0.0) {
      completeValue = 1;
    } else {
      completeValue = limit;
    }
    
  } else if (activationType == 2) { //sigmoid
    float4 sig = 1.0 / (1.0 + exp(-completeValue));
    completeValue = sig * (1 - sig);
    
  } else if (activationType == 3) { //swish
    completeValue = (exp(-completeValue) * (completeValue + 1) + 1) / pow((1 + exp(-completeValue)), 2);
    
  } else if (activationType == 4) { //tanH
    float4 denom = 1.0 + exp(-2 * completeValue);
    float4 tanActivate = (2.0 / denom) - 1.0;
    completeValue = 1 - (pow(tanActivate, 2));
    
  } else if (activationType == 5) { //none
    completeValue = 1;
  }
  
  outTexture.write(completeValue, coord, slice);
}

kernel void activation(texture2d_array<float, access::read> inTexture [[ texture(0) ]],
                       texture2d_array<float, access::write> outTexture [[ texture(1) ]],
                       const device uint& activationType [[ buffer(2) ]],
                       const device float& limit,
                       uint3 gid [[ thread_position_in_grid ]]) {
  
  uint2 coord = uint2(gid.x, gid.y);
  uint slice = gid.z;
  
  float4 completeValue = inTexture.read(coord, slice);
  
  if (activationType == 0) { //relu
    completeValue = max((float)0, completeValue);
    
  } else if (activationType == 1) { //leaky relu
    if (completeValue.x < 0.0 && completeValue.y < 0.0 && completeValue.z < 0.0 && completeValue.w < 0.0) {
      completeValue = limit * completeValue;
    } else {
      completeValue = completeValue;
    }
    
  } else if (activationType == 2) { //sigmoid
    completeValue = 1.0 / (1.0 + exp(-completeValue));
    
  } else if (activationType == 3) { //swish
    float4 sigmoid = 1.0 / (1.0 + exp(-completeValue));
    completeValue = completeValue * sigmoid;
    
  } else if (activationType == 4) { //tanH
    float4 denom = 1.0 + exp(-2 * completeValue);
    completeValue = (2.0 / denom) - 1.0;
    
  } else if (activationType == 5) { //none
    //results[resultIndex] = completeValue;
  }
  
  outTexture.write(completeValue, coord, slice);
}

kernel void conv2d(texture2d_array<float, access::read> inTexture [[ texture(0) ]],
                   texture2d_array<float, access::write> outTexture [[ texture(1) ]],
                   texture2d_array<float, access::read> filter [[ texture(2) ]],
                   constant uint3& inSize [[ buffer(1) ]],
                   constant uint3& outSize [[ buffer(2) ]],
                   constant uint2& kernelSize [[ buffer(3) ]],
                   constant uint2& stride [[buffer(4) ]],
                   constant int& padding [[ buffer(5) ]],
                   uint3 gid [[ thread_position_in_grid ]]) {
  
  uint2 coord = uint2(gid.x, gid.y);
  uint slice = gid.z;

  uint4 paddingCalc = padding_calc(stride, padding, kernelSize, uint2(inSize.x, inSize.y));
  
  uint paddingLeft = paddingCalc.z;
  uint paddingRight = paddingCalc.w;
  uint paddingTop = paddingCalc.x;
  uint paddingBottom = paddingCalc.y;
  
  uint inputRows = inSize.y;
  uint inputColumns = inSize.x;
  
  uint filterRows = kernelSize.y;
  uint filterColumns = kernelSize.x;
  
  if (coord.x >= outSize.x || coord.y >= outSize.y)
    return;
  
  uint padded_row_total = inputRows + paddingLeft + paddingRight;
  uint padded_col_total = inputColumns + paddingTop + paddingBottom;
  
  float4 sum = 0;
  for (uint i = 0; i < filterColumns; i++) {
    for (uint j = 0; j < filterRows; j++) {
      
      uint2 filter_coord = uint2(i, j);
      float4 current_filter = filter.read(filter_coord, slice);

      uint x = coord.x * stride.x + i;
      uint y = coord.y * stride.y + j;
      
      if (x >= 0 && x < padded_row_total && y >= 0 && y < padded_col_total) {
        sum += inTexture.read(uint2(x - paddingRight, y - paddingTop), slice) * current_filter;
      } else {
        sum += 0;
      }
    }
  }
  
  outTexture.write(sum, coord, slice);
}
