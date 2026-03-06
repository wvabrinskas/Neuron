//
//  NeuronKernels.metal
//  Neuron
//
//  Optimal Metal compute kernels for neural network operations on Apple Silicon.
//  Designed for MTLBuffer-backed tensors with .storageModeShared.
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Activation index mapping (align with Activation.index())
// 0=reLu, 1=leakyRelu, 2=sigmoid, 3=swish, 4=tanh, 5=softmax, 6=seLu, 7=geLu, 8=none

kernel void neuron_activation(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint& activationType [[buffer(2)]],
    constant float& leakyAlpha  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    float x = input[gid];
    float result = x;

    switch (activationType) {
        case 0: // reLu
            result = max(0.0f, x);
            break;
        case 1: // leakyRelu
            result = x > 0.0f ? x : leakyAlpha * x;
            break;
        case 2: // sigmoid
            result = 1.0f / (1.0f + exp(-x));
            break;
        case 3: // swish
            result = x / (1.0f + exp(-x));
            break;
        case 4: // tanh
            result = tanh(x);
            break;
        case 5: // softmax (identity in forward; softmax applied separately)
        case 8: // none
            result = x;
            break;
        case 6: { // seLu: lambda=1.0507, alpha=1.6733
            const float lambda = 1.0507f;
            const float alpha = 1.6733f;
            result = x > 0.0f ? lambda * x : lambda * alpha * (exp(x) - 1.0f);
            break;
        }
        case 7: { // geLu (tanh approximation)
            float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
            result = x * cdf;
            break;
        }
        default:
            result = x;
    }

    output[gid] = result;
}

kernel void neuron_derivate(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint& activationType [[buffer(2)]],
    constant float& leakyAlpha  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    float x = input[gid];
    float result = 1.0f;

    switch (activationType) {
        case 0: // reLu
            result = x >= 0.0f ? 1.0f : 0.0f;
            break;
        case 1: // leakyRelu
            result = x > 0.0f ? 1.0f : leakyAlpha;
            break;
        case 2: { // sigmoid: sig * (1 - sig)
            float sig = 1.0f / (1.0f + exp(-x));
            result = sig * (1.0f - sig);
            break;
        }
        case 3: { // swish derivative
            float sig = 1.0f / (1.0f + exp(-x));
            result = sig + x * sig * (1.0f - sig);
            break;
        }
        case 4: { // tanh: 1 - tanh^2
            float t = tanh(x);
            result = 1.0f - t * t;
            break;
        }
        case 5: // softmax
        case 8: // none
            result = 1.0f;
            break;
        case 6: { // seLu derivative
            const float lambda = 1.0507f;
            const float alpha = 1.6733f;
            result = x > 0.0f ? lambda : lambda * alpha * exp(x);
            break;
        }
        case 7: { // geLu derivative (tanh approximation)
            float tanhArg = 0.7978845608f * (x + 0.044715f * x * x * x);
            float tanhVal = tanh(tanhArg);
            float cdf = 0.5f * (1.0f + tanhVal);
            float gradTanh = 0.5f * (1.0f - tanhVal * tanhVal) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
            result = cdf + x * gradTanh;
            break;
        }
        default:
            result = 1.0f;
    }

    output[gid] = result;
}

// MARK: - Matrix multiplication (tiled 16x16)

#define TILE_SIZE 16

kernel void neuron_matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + lid.y;
    uint col = tgid.x * TILE_SIZE + lid.x;

    float sum = 0.0f;

    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        uint aCol = t * TILE_SIZE + lid.x;
        uint bRow = t * TILE_SIZE + lid.y;

        tileA[lid.y][lid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[lid.y][lid.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += tileA[lid.y][i] * tileB[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// MARK: - Convolution

struct Conv2DParams {
    uint N;
    uint C;
    uint H;
    uint W;
    uint K;
    uint kH;
    uint kW;
    uint oH;
    uint oW;
    uint strideH;
    uint strideW;
    uint padH;
    uint padW;
    uint hasBias;  // 1 if bias buffer is valid, 0 otherwise
};

kernel void neuron_conv2d_implicit_gemm(
    device const float* input   [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant Conv2DParams& p    [[buffer(3)]],
    device const float* bias    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint spatial = gid.y;
    uint k = gid.x;

    uint total_spatial = p.N * p.oH * p.oW;
    if (spatial >= total_spatial || k >= p.K) return;

    uint n   = spatial / (p.oH * p.oW);
    uint rem = spatial % (p.oH * p.oW);
    uint oh  = rem / p.oW;
    uint ow  = rem % p.oW;

    float sum = 0.0f;
    uint gemm_K = p.C * p.kH * p.kW;

    for (uint ck = 0; ck < gemm_K; ck++) {
        uint c  = ck / (p.kH * p.kW);
        uint kr = (ck % (p.kH * p.kW)) / p.kW;
        uint kc = ck % p.kW;

        int ih = (int)(oh * p.strideH + kr) - (int)p.padH;
        int iw = (int)(ow * p.strideW + kc) - (int)p.padW;

        float a = 0.0f;
        if (ih >= 0 && ih < (int)p.H && iw >= 0 && iw < (int)p.W) {
            a = input[n * p.C * p.H * p.W + c * p.H * p.W + ih * p.W + iw];
        }

        float w = weights[k * gemm_K + ck];
        sum += a * w;
    }

    if (p.hasBias) {
        sum += bias[k];
    }

    output[n * p.K * p.oH * p.oW + k * p.oH * p.oW + oh * p.oW + ow] = sum;
}

kernel void neuron_conv2d_backward_input(
    device const float* grad_output [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device float* grad_input        [[buffer(2)]],
    constant Conv2DParams& p        [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = p.N * p.C * p.H * p.W;
    if (gid >= total) return;

    uint n  = gid / (p.C * p.H * p.W);
    uint c  = (gid / (p.H * p.W)) % p.C;
    uint h  = (gid / p.W) % p.H;
    uint w  = gid % p.W;

    float sum = 0.0f;

    for (uint k = 0; k < p.K; k++) {
        for (uint kh = 0; kh < p.kH; kh++) {
            for (uint kw = 0; kw < p.kW; kw++) {
                int oh_check = (int)h + (int)p.padH - (int)kh;
                int ow_check = (int)w + (int)p.padW - (int)kw;

                if (oh_check % (int)p.strideH != 0) continue;
                if (ow_check % (int)p.strideW != 0) continue;

                int oh = oh_check / (int)p.strideH;
                int ow = ow_check / (int)p.strideW;

                if (oh >= 0 && oh < (int)p.oH && ow >= 0 && ow < (int)p.oW) {
                    float go = grad_output[n * p.K * p.oH * p.oW + k * p.oH * p.oW + oh * p.oW + ow];
                    float wt = weights[k * p.C * p.kH * p.kW + c * p.kH * p.kW + kh * p.kW + kw];
                    sum += go * wt;
                }
            }
        }
    }

    grad_input[gid] = sum;
}

kernel void neuron_conv2d_backward_weights(
    device const float* input       [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_weights      [[buffer(2)]],
    constant Conv2DParams& p        [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint total_weights = p.K * p.C * p.kH * p.kW;
    if (gid >= total_weights) return;

    uint k  = gid / (p.C * p.kH * p.kW);
    uint c  = (gid / (p.kH * p.kW)) % p.C;
    uint kh = (gid / p.kW) % p.kH;
    uint kw = gid % p.kW;

    float sum = 0.0f;

    for (uint n = 0; n < p.N; n++) {
        for (uint oh = 0; oh < p.oH; oh++) {
            for (uint ow = 0; ow < p.oW; ow++) {
                int ih = (int)(oh * p.strideH + kh) - (int)p.padH;
                int iw = (int)(ow * p.strideW + kw) - (int)p.padW;

                if (ih >= 0 && ih < (int)p.H && iw >= 0 && iw < (int)p.W) {
                    float inp = input[n * p.C * p.H * p.W + c * p.H * p.W + ih * p.W + iw];
                    float go  = grad_output[n * p.K * p.oH * p.oW + k * p.oH * p.oW + oh * p.oW + ow];
                    sum += inp * go;
                }
            }
        }
    }

    grad_weights[gid] = sum;
}

// MARK: - Max Pool 2x2
// Input [N,C,H,W], output [N,C,outH,outW] where outH=(H+1)/2, outW=(W+1)/2.
// Indices: uint32 per output element, 0-3 for which of the 4 input positions had the max.
// Order: (r,c)=0, (r+1,c)=1, (r,c+1)=2, (r+1,c+1)=3.

kernel void neuron_max_pool_2x2(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    device uint* indices       [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    constant uint& C            [[buffer(4)]],
    constant uint& H            [[buffer(5)]],
    constant uint& W            [[buffer(6)]],
    constant uint& outH         [[buffer(7)]],
    constant uint& outW         [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = N * C * outH * outW;
    if (gid >= total) return;

    uint n = gid / (C * outH * outW);
    uint c = (gid / (outH * outW)) % C;
    uint oh = (gid / outW) % outH;
    uint ow = gid % outW;

    uint ih = oh * 2;
    uint iw = ow * 2;

    uint base = n * C * H * W + c * H * W;

    float a = (ih < H && iw < W) ? input[base + ih * W + iw] : 0.0f;
    float b = (ih + 1 < H && iw < W) ? input[base + (ih + 1) * W + iw] : 0.0f;
    float c0 = (ih < H && iw + 1 < W) ? input[base + ih * W + (iw + 1)] : 0.0f;
    float d = (ih + 1 < H && iw + 1 < W) ? input[base + (ih + 1) * W + (iw + 1)] : 0.0f;

    float maxVal = max(max(max(a, b), c0), d);
    uint idx = 0;
    if (a == maxVal) idx = 0;
    else if (b == maxVal) idx = 1;
    else if (c0 == maxVal) idx = 2;
    else idx = 3;

    output[gid] = maxVal;
    indices[gid] = idx;
}

// MARK: - Instance Normalization
// Input/output [N,C,H,W] in NCHW. Gamma, beta [C]. Normalize per (n,c) over H*W.

kernel void neuron_instance_norm(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    device const float* gamma   [[buffer(2)]],
    device const float* beta    [[buffer(3)]],
    constant uint& N             [[buffer(4)]],
    constant uint& C             [[buffer(5)]],
    constant uint& H             [[buffer(6)]],
    constant uint& W             [[buffer(7)]],
    constant float& epsilon      [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    uint nc = N * C;
    if (gid >= nc) return;

    uint n = gid / C;
    uint c = gid % C;

    uint spatial = H * W;
    uint base = (n * C + c) * spatial;

    float sum = 0.0f;
    for (uint i = 0; i < spatial; i++) {
        sum += input[base + i];
    }
    float mean = sum / float(spatial);

    float sumSq = 0.0f;
    for (uint i = 0; i < spatial; i++) {
        float d = input[base + i] - mean;
        sumSq += d * d;
    }
    float variance = sumSq / float(spatial);
    float std = sqrt(variance + epsilon);

    float g = gamma[c];
    float b = beta[c];

    for (uint i = 0; i < spatial; i++) {
        float x = (input[base + i] - mean) / std;
        output[base + i] = g * x + b;
    }
}

kernel void neuron_conv_transpose2d(
    device const float* input   [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant Conv2DParams& p    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = p.N * p.K * p.oH * p.oW;
    if (gid >= total) return;

    uint n   = gid / (p.K * p.oH * p.oW);
    uint k   = (gid / (p.oH * p.oW)) % p.K;
    uint oh  = (gid / p.oW) % p.oH;
    uint ow  = gid % p.oW;

    float sum = 0.0f;

    for (uint c = 0; c < p.C; c++) {
        for (uint kh = 0; kh < p.kH; kh++) {
            for (uint kw = 0; kw < p.kW; kw++) {
                int ih_check = (int)oh + (int)p.padH - (int)kh;
                int iw_check = (int)ow + (int)p.padW - (int)kw;

                if (ih_check % (int)p.strideH != 0) continue;
                if (iw_check % (int)p.strideW != 0) continue;

                int ih = ih_check / (int)p.strideH;
                int iw = iw_check / (int)p.strideW;

                if (ih >= 0 && ih < (int)p.H && iw >= 0 && iw < (int)p.W) {
                    float inp = input[n * p.C * p.H * p.W + c * p.H * p.W + ih * p.W + iw];
                    float wt = weights[c * p.K * p.kH * p.kW + k * p.kH * p.kW + kh * p.kW + kw];
                    sum += wt * inp;
                }
            }
        }
    }

    output[gid] = sum;
}
