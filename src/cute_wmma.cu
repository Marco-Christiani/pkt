#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

template <int TM, int TN, int TK>
__global__ void mlp_kernel(
    int B, int In, int Hdim1, int Out,
    __nv_bfloat16* X, int ldX,
    __nv_bfloat16* W0, int ldW0,
    __nv_bfloat16* W1, int ldW1,
    __nv_bfloat16* H1, int ldH1,
    float* Y, int ldY
) {
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    if (warpId > 0) return;

    int m0 = blockIdx.x * TM;
    if (m0 >= B) return;

    __shared__ float smem[WMMA_M][128];  // Assuming Hdim1 <= 128

    // Process H1 = ReLU(X @ W0^T) - store to shared memory
    for (int n0 = 0; n0 < Hdim1; n0 += WMMA_N) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        for (int k = 0; k < In; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, X + m0 * ldX + k, ldX);
            wmma::load_matrix_sync(b_frag, W0 + n0 * ldW0 + k, ldW0);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Apply ReLU
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = c_frag.x[i] > 0.0f ? c_frag.x[i] : 0.0f;
        }

        // gbm
        wmma::store_matrix_sync(&smem[0][n0], c_frag, 128, wmma::mem_row_major);
    }

    __syncthreads(); 

    // write h1 to gbm for debgging
    for (int i = threadIdx.x; i < WMMA_M * Hdim1; i += blockDim.x) {
        int row = i / Hdim1;
        int col = i % Hdim1;
        if (m0 + row < B && col < Hdim1) {
            H1[(m0 + row) * ldH1 + col] = __float2bfloat16(smem[row][col]);
        }
    }
    // Process Y = H1 @ W1^T - load from smem with conversion
    for (int n0 = 0; n0 < Out; n0 += WMMA_N) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        for (int k = 0; k < Hdim1; k += WMMA_K) {
            // convert 16x16 tile from float smem to bf16 for loading
            __nv_bfloat16 temp[WMMA_M * WMMA_K];
            for (int i = threadIdx.x; i < WMMA_M * WMMA_K; i += blockDim.x) {
                int row = i / WMMA_K;
                int col = i % WMMA_K;
                temp[i] = __float2bfloat16(smem[row][k + col]);
            }
            
            wmma::load_matrix_sync(a_frag, temp, WMMA_K);
            wmma::load_matrix_sync(b_frag, W1 + n0 * ldW1 + k, ldW1);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        wmma::store_matrix_sync(Y + m0 * ldY + n0, c_frag, ldY, wmma::mem_row_major);
    }
}

int main() {
    const int B = 16, In = 32, Hdim1 = 64, Out = 16;

    __nv_bfloat16 *hX = new __nv_bfloat16[B*In];
    __nv_bfloat16 *hW0 = new __nv_bfloat16[Hdim1*In];
    __nv_bfloat16 *hW1 = new __nv_bfloat16[Out*Hdim1];
    __nv_bfloat16 *hH1 = new __nv_bfloat16[B*Hdim1];
    float *hY = new float[B*Out];

    // Initialize
    for (int i = 0; i < B*In; i++) hX[i] = __float2bfloat16(1.0f);
    for (int i = 0; i < Hdim1*In; i++) hW0[i] = __float2bfloat16(0.1f);
    for (int i = 0; i < Out*Hdim1; i++) hW1[i] = __float2bfloat16(0.1f);

    __nv_bfloat16 *dX, *dW0, *dW1, *dH1;
    float *dY;

    cudaMalloc(&dX, B*In*sizeof(__nv_bfloat16));
    cudaMalloc(&dW0, Hdim1*In*sizeof(__nv_bfloat16));
    cudaMalloc(&dW1, Out*Hdim1*sizeof(__nv_bfloat16));
    cudaMalloc(&dH1, B*Hdim1*sizeof(__nv_bfloat16));  // Changed to bf16
    cudaMalloc(&dY, B*Out*sizeof(float));

    cudaMemcpy(dX, hX, B*In*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(dW0, hW0, Hdim1*In*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(dW1, hW1, Out*Hdim1*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    dim3 grid(B/16);
    dim3 block(32);

    mlp_kernel<16,16,16><<<grid, block>>>(B, In, Hdim1, Out, dX, In, dW0, In, dW1, Hdim1, dH1, Hdim1, dY, Out);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(hH1, dH1, B*Hdim1*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(hY, dY, B*Out*sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("H1[0] = %.4f (should be ~3.2)\n", __bfloat162float(hH1[0]));
    printf("Y[0] = %.4f\n", hY[0]);
cudaMemcpy(hY, dY, B*Out*sizeof(float), cudaMemcpyDeviceToHost);

    printf("H1[0] = %.4f (should be ~3.2)\n", __bfloat162float(hH1[0]));
    printf("H1[1] = %.4f\n", __bfloat162float(hH1[1]));
    printf("Y[0] = %.4f\n", hY[0]);
    printf("Y[1] = %.4f\n", hY[1]);

    delete[] hX; delete[] hW0; delete[] hW1; delete[] hH1; delete[] hY;
    cudaFree(dX); cudaFree(dW0); cudaFree(dW1); cudaFree(dH1); cudaFree(dY);
    return 0;
}
