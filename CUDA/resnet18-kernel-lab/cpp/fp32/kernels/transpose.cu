// transpose.cu : out[I*B] = in[B*I] (row-major)
extern "C" __global__
void transpose_bi_to_ib(float* __restrict__ out, const float* __restrict__ in,
                        int B, int I)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..I-1
    int b = blockIdx.y * blockDim.y + threadIdx.y; // 0..B-1
    if (i < I && b < B) {
        out[i*B + b] = in[b*I + i];
    }
}
