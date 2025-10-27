// cpp/fp32/runtime/infer_stem_block0.cpp

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "utils.hpp"

// Step2 커널들
extern "C" __global__
void im2col_nchw(const float*,int,int,int,int,int,int,int,int,int,int,float*);
extern "C" __global__
void sgemm_tiled(const float*,const float*,float*,int,int,int);
extern "C" __global__
void bn_inference(float*,const float*,const float*,const float*,const float*,float,int,int,int);
extern "C" __global__
void relu_forward(float*,int);

// Step3 커널들
extern "C" __global__
void maxpool2d_kernel_fp32(
    const float*, float*,
    int,int,int,int,
    int,int,int,int,int,int,int,int);
extern "C"
void basic_block_fp32_forward_identity(
    const float* d_x,
    float* d_out,
    float* d_tmp1,
    float* d_tmp2,
    float* d_colBuf,
    const float* d_w1col,
    int N, int C, int H, int W,
    int C1out,
    int k1H,int k1W,
    int s1H,int s1W,
    int p1H,int p1W,
    const float* d_bn1_gamma,
    const float* d_bn1_beta,
    const float* d_bn1_mean,
    const float* d_bn1_var,
    float bn1_eps,
    const float* d_w2col,
    int C2out,
    int k2H,int k2W,
    int s2H,int s2W,
    int p2H,int p2W,
    const float* d_bn2_gamma,
    const float* d_bn2_beta,
    const float* d_bn2_mean,
    const float* d_bn2_var,
    float bn2_eps,
    cudaStream_t stream
);

// Timer는 Step2에서 사용한 동일 구조라고 가정
static void usage(){
    std::cout<<"--manifest exports/resnet18/fp32 --input input.bin --expect_block0 block0_out.bin\n";
}

int main(int argc, char** argv){
    std::string mani, inputPath, expectPath;
    for (int i=1;i<argc;i++){
        std::string a=argv[i];
        if (a=="--manifest" && i+1<argc) mani=argv[++i];
        else if (a=="--input" && i+1<argc) inputPath=argv[++i];
        else if (a=="--expect" && i+1<argc) expectPath=argv[++i];
    }
    if (mani.empty()||inputPath.empty()||expectPath.empty()){ usage(); return 1; }

    // --------------------------
    // 1. shape 정의
    // --------------------------
    // stem input
    int N=1, C_in=3, H_in=224, W_in=224;

    // conv1 (7x7 s2 p3) -> (1,64,112,112)
    int C1_out=64, k1H=7,k1W=7, s1H=2,s1W=2, p1H=3,p1W=3;
    int H1 = (H_in+2*p1H-k1H)/s1H + 1; //112
    int W1 = (W_in+2*p1W-k1W)/s1W + 1; //112

    // maxpool (3x3 s2 p1) -> (1,64,56,56)
    int poolKH=3,poolKW=3;
    int poolSH=2,poolSW=2;
    int poolPH=1,poolPW=1;
    int H2 = (H1+2*poolPH-poolKH)/poolSH + 1; //56
    int W2 = (W1+2*poolPW-poolKW)/poolSW + 1; //56

    // block0 (3x3 s1 p1 twice, C stays 64)
    int blk_kH=3, blk_kW=3;
    int blk_sH=1, blk_sW=1;
    int blk_pH=1, blk_pW=1;
    int Cblk = 64; // stays 64, output shape (1,64,56,56)

    // --------------------------
    // 2. 호스트에서 파라미터/입력 로드
    // --------------------------
    // conv1 weights: conv1.weight.bin -> [64,3,7,7]
    // reshape to Wcol: [64, 3*7*7]
    // bn1 params: bn1.weight.bin, bn1.bias.bin, bn1.running_mean.bin, bn1.running_var.bin
    //
    // block0 conv1,conv2 weights:
    //   layer1.0.conv1.weight.bin → [64,64,3,3] -> flatten to [64, 64*3*3]
    //   layer1.0.conv2.weight.bin → same
    // block0 bn params:
    //   layer1.0.bn1.(weight,bias,mean,var).bin
    //   layer1.0.bn2.(weight,bias,mean,var).bin
    //
    // expectPath = block0 output reference from PyTorch after stem+block0.
    //
    auto x_input = load_bin_f32(inputPath,  N*C_in*H_in*W_in);

    // stem conv1/bn1 params
    auto w_conv1 = load_bin_f32(mani+"/conv1.weight.bin", C1_out*C_in*k1H*k1W);
    auto bn1_w   = load_bin_f32(mani+"/bn1.weight.bin",   C1_out);
    auto bn1_b   = load_bin_f32(mani+"/bn1.bias.bin",     C1_out);
    auto bn1_m   = load_bin_f32(mani+"/bn1.running_mean.bin", C1_out);
    auto bn1_v   = load_bin_f32(mani+"/bn1.running_var.bin",  C1_out);

    // block0 conv/bn params
    auto w_blk1  = load_bin_f32(mani+"/layer1.0.conv1.weight.bin", Cblk*Cblk*blk_kH*blk_kW);
    auto w_blk2  = load_bin_f32(mani+"/layer1.0.conv2.weight.bin", Cblk*Cblk*blk_kH*blk_kW);

    auto blk1_w  = load_bin_f32(mani+"/layer1.0.bn1.weight.bin", Cblk);
    auto blk1_b  = load_bin_f32(mani+"/layer1.0.bn1.bias.bin",   Cblk);
    auto blk1_m  = load_bin_f32(mani+"/layer1.0.bn1.running_mean.bin", Cblk);
    auto blk1_v  = load_bin_f32(mani+"/layer1.0.bn1.running_var.bin",  Cblk);

    auto blk2_w  = load_bin_f32(mani+"/layer1.0.bn2.weight.bin", Cblk);
    auto blk2_b  = load_bin_f32(mani+"/layer1.0.bn2.bias.bin",   Cblk);
    auto blk2_m  = load_bin_f32(mani+"/layer1.0.bn2.running_mean.bin", Cblk);
    auto blk2_v  = load_bin_f32(mani+"/layer1.0.bn2.running_var.bin",  Cblk);

    auto y_expect = load_bin_f32(expectPath, N*Cblk*H2*W2); // torch ref after block0

    // conv1 weight -> Wcol(flat)
    std::vector<float> Wcol_conv1(C1_out * (C_in*k1H*k1W));
    {
        for (int oc=0; oc<C1_out; ++oc){
            for (int ic=0; ic<C_in; ++ic){
                for (int kh=0; kh<k1H; ++kh){
                    for (int kw=0; kw<k1W; ++kw){
                        int r   = ic*k1H*k1W + kh*k1W + kw; // row idx in KCOL
                        int src = oc*C_in*k1H*k1W + ic*k1H*k1W + kh*k1W + kw;
                        Wcol_conv1[oc*(C_in*k1H*k1W) + r] = w_conv1[src];
                    }
                }
            }
        }
    }

    // block0 conv1 weight -> flatten [64, 64*3*3]
    std::vector<float> Wcol_blk1(Cblk * (Cblk*blk_kH*blk_kW));
    {
        for (int oc=0; oc<Cblk; ++oc){
            for (int ic=0; ic<Cblk; ++ic){
                for (int kh=0; kh<blk_kH; ++kh){
                    for (int kw=0; kw<blk_kW; ++kw){
                        int r   = ic*blk_kH*blk_kW + kh*blk_kW + kw;
                        int src = oc*Cblk*blk_kH*blk_kW + ic*blk_kH*blk_kW + kh*blk_kW + kw;
                        Wcol_blk1[oc*(Cblk*blk_kH*blk_kW) + r] = w_blk1[src];
                    }
                }
            }
        }
    }

    // block0 conv2 weight -> flatten
    std::vector<float> Wcol_blk2 = Wcol_blk1; // if conv2 has same shape as conv1
    // 만약 layer1.0.conv2.weight.bin이 다르면 위랑 똑같이 변환해서 Wcol_blk2 채우면 됨.
    {
        for (int oc=0; oc<Cblk; ++oc){
            for (int ic=0; ic<Cblk; ++ic){
                for (int kh=0; kh<blk_kH; ++kh){
                    for (int kw=0; kw<blk_kW; ++kw){
                        int r   = ic*blk_kH*blk_kW + kh*blk_kW + kw;
                        int src = oc*Cblk*blk_kH*blk_kW + ic*blk_kH*blk_kW + kh*blk_kW + kw;
                        Wcol_blk2[oc*(Cblk*blk_kH*blk_kW) + r] = w_blk2[src];
                    }
                }
            }
        }
    }

    // --------------------------
    // 3. 디바이스 메모리 준비
    // --------------------------
    float *dInput, *dCol, *dConv1Out, *dPoolOut;
    float *dBlockOut, *dTmp1, *dTmp2;

    CUDA_CHECK(cudaMalloc(&dInput,    N*C_in*H_in*W_in*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCol,      (C_in*k1H*k1W)* (H1*W1) *sizeof(float))); // stem conv1 use
    CUDA_CHECK(cudaMalloc(&dConv1Out, C1_out*H1*W1*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPoolOut,  C1_out*H2*W2*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dBlockOut, Cblk*H2*W2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTmp1,     Cblk*H2*W2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTmp2,     Cblk*H2*W2*sizeof(float)));

    // block col buf (for convs inside block0)
    float *dColBlock;
    CUDA_CHECK(cudaMalloc(&dColBlock, (Cblk*blk_kH*blk_kW)*(H2*W2)*sizeof(float)));

    float *dWconv1, *dBn1W,*dBn1B,*dBn1M,*dBn1V;
    CUDA_CHECK(cudaMalloc(&dWconv1, Wcol_conv1.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn1W,  C1_out*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn1B,  C1_out*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn1M,  C1_out*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn1V,  C1_out*sizeof(float)));

    float *dWblk1,*dWblk2;
    float *dBlk1W,*dBlk1B,*dBlk1M,*dBlk1V;
    float *dBlk2W,*dBlk2B,*dBlk2M,*dBlk2V;
    CUDA_CHECK(cudaMalloc(&dWblk1, Wcol_blk1.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWblk2, Wcol_blk2.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBlk1W, Cblk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBlk1B, Cblk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBlk1M, Cblk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBlk1V, Cblk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBlk2W, Cblk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBlk2B, Cblk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBlk2M, Cblk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBlk2V, Cblk*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dInput,   x_input.data(), x_input.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWconv1,  Wcol_conv1.data(), Wcol_conv1.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn1W,    bn1_w.data(),     C1_out*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn1B,    bn1_b.data(),     C1_out*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn1M,    bn1_m.data(),     C1_out*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn1V,    bn1_v.data(),     C1_out*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dWblk1,   Wcol_blk1.data(), Wcol_blk1.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWblk2,   Wcol_blk2.data(), Wcol_blk2.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBlk1W,   blk1_w.data(),    Cblk*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBlk1B,   blk1_b.data(),    Cblk*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBlk1M,   blk1_m.data(),    Cblk*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBlk1V,   blk1_v.data(),    Cblk*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBlk2W,   blk2_w.data(),    Cblk*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBlk2B,   blk2_b.data(),    Cblk*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBlk2M,   blk2_m.data(),    Cblk*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBlk2V,   blk2_v.data(),    Cblk*sizeof(float), cudaMemcpyHostToDevice));

    // --------------------------
    // 4. 실행 + 타이밍
    // --------------------------
    Timer T;
    float ms_conv1, ms_bn1, ms_relu1, ms_pool, ms_block;

    // 4-1. stem conv1 (im2col + sgemm)
    {
        int KCOL  = C_in*k1H*k1W;
        int NCOL  = H1*W1;

        dim3 blkIm2(16,16);
        dim3 grdIm2((W1+blkIm2.x-1)/blkIm2.x,
                    (H1+blkIm2.y-1)/blkIm2.y);

        T.start();
        im2col_nchw<<<grdIm2, blkIm2>>>(dInput,
            N,C_in,H_in,W_in,
            k1H,k1W,
            s1H,s1W,
            p1H,p1W,
            dCol);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_conv1 = T.stop();

        dim3 blkG(32,32);
        dim3 grdG((NCOL+blkG.x-1)/blkG.x,
                  (C1_out+blkG.y-1)/blkG.y);

        T.start();
        sgemm_tiled<<<grdG, blkG>>>(dWconv1, dCol, dConv1Out,
            C1_out, NCOL, KCOL);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_conv1 += T.stop(); // add GEMM time
    }

    // 4-2. stem bn1
    {
        int total_spatial = C1_out*H1*W1;
        dim3 blkBN(256);
        dim3 grdBN((total_spatial+blkBN.x-1)/blkBN.x);

        T.start();
        bn_inference<<<grdBN, blkBN>>>(dConv1Out,
            dBn1W, dBn1B, dBn1M, dBn1V,
            1e-5f,
            C1_out, H1, W1);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_bn1 = T.stop();
    }

    // 4-3. stem relu1
    {
        int total_relu = C1_out*H1*W1;
        dim3 blkR(256);
        dim3 grdR((total_relu+blkR.x-1)/blkR.x);

        T.start();
        relu_forward<<<grdR, blkR>>>(dConv1Out, total_relu);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_relu1 = T.stop();
    }

    // 4-4. maxpool
    {
        int total_pool = C1_out*H2*W2;
        dim3 blkP(256);
        dim3 grdP((total_pool+blkP.x-1)/blkP.x);

        T.start();
        maxpool2d_kernel_fp32<<<grdP, blkP>>>(
            dConv1Out, dPoolOut,
            N, C1_out, H1, W1,
            poolKH, poolKW,
            poolSH, poolSW,
            poolPH, poolPW,
            H2, W2
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_pool = T.stop();
    }

    // 4-5. block0
    {
        T.start();
        basic_block_fp32_forward_identity(
            dPoolOut,     // x
            dBlockOut,    // out
            dTmp1,        // tmp1
            dTmp2,        // tmp2
            dColBlock,    // im2col workspace for block

            dWblk1,       // conv1 weights (flattened)
            N, Cblk, H2, W2,
            Cblk,         // conv1 outC (still 64)
            blk_kH, blk_kW,
            blk_sH, blk_sW,
            blk_pH, blk_pW,
            dBlk1W, dBlk1B, dBlk1M, dBlk1V,
            1e-5f,

            dWblk2,       // conv2 weights
            Cblk,         // conv2 outC
            blk_kH, blk_kW,
            blk_sH, blk_sW,
            blk_pH, blk_pW,
            dBlk2W, dBlk2B, dBlk2M, dBlk2V,
            1e-5f,

            0 // stream
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_block = T.stop();
    }

    // --------------------------
    // 5. 결과 비교
    // --------------------------
    std::vector<float> y_out(N*Cblk*H2*W2);
    CUDA_CHECK(cudaMemcpy(y_out.data(), dBlockOut, y_out.size()*sizeof(float),
                          cudaMemcpyDeviceToHost));

    double max_abs=0.0, mean_abs=0.0;
    for (size_t i=0;i<y_out.size();++i){
        double d = std::fabs((double)y_out[i] - (double)y_expect[i]);
        if (d > max_abs) max_abs = d;
        mean_abs += d;
    }
    mean_abs /= y_out.size();

    std::cout<<"Step3 stem+block0 forward done\n";
    std::cout<<"  conv1 (im2col+gemm): "<<ms_conv1<<" ms\n";
    std::cout<<"  bn1                : "<<ms_bn1<<" ms\n";
    std::cout<<"  relu1              : "<<ms_relu1<<" ms\n";
    std::cout<<"  maxpool            : "<<ms_pool<<" ms\n";
    std::cout<<"  block0             : "<<ms_block<<" ms\n";
    std::cout<<"Diff block0 out vs ref: max_abs="<<max_abs
             <<" mean_abs="<<mean_abs<<"\n";

    if (max_abs <= 1e-4) {
        std::cout<<"[OK] within atol 1e-4\n";
    } else {
        std::cerr<<"[WARN] exceed atol 1e-4\n";
    }

    // --------------------------
    // 6. cleanup
    // --------------------------
    cudaFree(dInput);
    cudaFree(dCol);
    cudaFree(dConv1Out);
    cudaFree(dPoolOut);
    cudaFree(dBlockOut);
    cudaFree(dTmp1);
    cudaFree(dTmp2);
    cudaFree(dColBlock);

    cudaFree(dWconv1);
    cudaFree(dBn1W); cudaFree(dBn1B); cudaFree(dBn1M); cudaFree(dBn1V);

    cudaFree(dWblk1); cudaFree(dWblk2);
    cudaFree(dBlk1W); cudaFree(dBlk1B); cudaFree(dBlk1M); cudaFree(dBlk1V);
    cudaFree(dBlk2W); cudaFree(dBlk2B); cudaFree(dBlk2M); cudaFree(dBlk2V);

    return 0;
}
