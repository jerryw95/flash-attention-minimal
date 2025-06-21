#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Warp Reduce Max
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}


__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    int row = tx / Br;
    int col = tx % Br;

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int kv_tile_size = Bc * d;  // size of Kj, Vj
    int q_tile_size = Br * d;  // size of Qi
    float* Qi = sram;
    float* Kj = &sram[q_tile_size];
    float* Vj = &sram[q_tile_size + kv_tile_size];
    float* S = &sram[q_tile_size + 2 * kv_tile_size];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < 2; x++) {
            Kj[tx + Br*Br*x] = K[qkv_offset + (kv_tile_size * j) + tx + Br*Br*x];
            Vj[tx + Br*Br*x] = V[qkv_offset + (kv_tile_size * j) + tx + Br*Br*x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < 2; x++) {
                Qi[tx + Br*Br*x] = Q[qkv_offset + (q_tile_size * i) + tx + Br*Br*x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + row];
            float row_l_prev = l[lm_offset + (Br * i) + row];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;

            float sum = 0;
            for (int x = 0; x < d; x++) {
                sum += Qi[(row * d) + x] * Kj[(col * d) + x];
            }
            sum *= softmax_scale;
            S[(Bc * row) + col] = sum;

            row_m = warp_reduce_max_f32<WARP_SIZE>(sum);

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            float exp_val = __expf(S[(Bc * row) + col] - row_m);
            S[(Bc * row) + col] = exp_val;
            row_l = warp_reduce_sum_f32<WARP_SIZE>(exp_val);

            // Compute new m and l
            float row_m_new = fmaxf(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < 2; x++) {
                float pv = 0;  // Pij * Vj
                int col_offset = Br * x;
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * row) + y] * Vj[(y * d) + col + col_offset];
                }
                O[qkv_offset + (q_tile_size * i) + (row * d) + col + col_offset] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (q_tile_size * i) + (row * d) + col + col_offset]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            if(col == 0) {
                // Write l and m to HBM only for the first column
                m[lm_offset + (Br * i) + row] = row_m_new;
                l[lm_offset + (Br * i) + row] = row_l_new;
            }
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    O = O.to(device); l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    int col_tile_size = Bc * d;  // size of Kj, Vj
    int row_tile_size = Br * d;  // size of Qi
    const int sram_size =
        (2 * col_tile_size * sizeof(float))  // SRAM size for Kj, Vj
        + (row_tile_size * sizeof(float))  // SRAM size for Qi
        + (Bc * Br * sizeof(float));  // SRAM size for S
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(1024);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}