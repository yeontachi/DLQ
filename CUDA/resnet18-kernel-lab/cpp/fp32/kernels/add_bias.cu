// add_bias.cu : C[O×B] += bias[O] (broadcast along B)
extern "C" __global__
void add_bias_OB(float* __restrict__ C, const float* __restrict__ bias,
                 int O, int B)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x; // 0..O-1
    int b = blockIdx.y * blockDim.y + threadIdx.y; // 0..B-1
    if (o < O && b < B) {
        C[o*B + b] += bias[o];
    }
}
