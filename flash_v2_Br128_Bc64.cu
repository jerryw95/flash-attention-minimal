#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])

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
void flash_attention_2_forward_kernel(
    float* Q,
    float* K,
    float* V,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    // float* L,
    float* O
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    int block_width = 8;
    int row = tx / block_width; // row from 0 to 128
    int col = tx % block_width; // col from 0 to 7

    // Offset into Q,K,V,O - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for L

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int kv_tile_size = Bc * d;  // size of Kj, Vj
    int q_tile_size = Br * d;  // size of Qi
    float* Qi = sram;
    float* KVj = &sram[q_tile_size];
    // float* Vj = &sram[q_tile_size + kv_tile_size];
    float* S = &sram[q_tile_size + kv_tile_size];

    for (int i = 0; i < Tr; ++i) {
        // if (i * Br + row >= N)
        //     break;  // break if we are done with the sequence

        // Load Qi from HBM to SRAM, l and m to registers
        // TODO: Add vectorized loading from DRAM //
        // for (int x = 0; x < 4; x++) {
        //     Qi[tx + Bc * Bc * x] = Q[qkv_offset + (q_tile_size * i) + tx + Bc * Bc * x];
        // }
        FLOAT4(Qi[row * d + col * 8]) = FLOAT4(Q[qkv_offset + (q_tile_size * i) + row * d + col * 8]);
        FLOAT4(Qi[row * d + col * 8 + 4]) = FLOAT4(Q[qkv_offset + (q_tile_size * i) + row * d + col * 8 + 4]);

        float row_m_prev = -INFINITY;
        float row_l_prev = 0;

        // Causal mask: j <= i
        for (int j = 0; j < Tc; ++j) {
            int new_row = tx / 16; // row from 0 to 64
            int new_col = tx % 16; // col from 0 to 15
            __syncthreads();
            // Load Kj Vj from HBM to SRAM
            FLOAT4(KVj[new_row * d + new_col * 4]) = FLOAT4(K[qkv_offset + (kv_tile_size * j) + new_row * d + new_col * 4]);
            // FLOAT4(Vj[row * d + col * 4]) = FLOAT4(V[qkv_offset + (kv_tile_size * j) + row * d + col * 4]);
            __syncthreads();
            // S_i^j = softmax_scale * QiKj^T
            // S_i^j[tx][y] = softmax_scale * Sum_{x = 0}^{d-1} Qi[tx][x] * Kj[y][x]
            float row_m = -INFINITY;
            for (int y = 0; y < Bc/block_width; y++) {
                //if (j * Bc + y >= N)
                //    break;  // break if we are done with the sequence
                //if (i * Br + tx < j * Bc + y)
                //    break;
                int col_offset = y * block_width;
                float val = 0;
                for (int x = 0; x < d; x++)
                    val += Qi[(row * d) + x] * KVj[((col+col_offset) * d) + x];
                val *= softmax_scale;
                S[(row * Bc) + col + col_offset ] = val;
                // Find the maximum value in the row S_i^j
                float warp_m =  warp_reduce_max_f32<WARP_SIZE/4>(val);
                row_m = fmaxf(row_m, warp_m);
            }

            __syncthreads();
            // FLOAT4(KVj[row * d + col * 4]) = FLOAT4(K[qkv_offset + (kv_tile_size * j) + row * d + col * 4]);
            FLOAT4(KVj[new_row * d + new_col * 4]) = FLOAT4(V[qkv_offset + (kv_tile_size * j) + new_row * d + new_col * 4]);
            __syncthreads();

            // m_i^j = max(m_i^j-1, row_max(S_i^j))
            float new_row_m = fmaxf(row_m_prev, row_m);

            // P_i^j = exp(S_i^j - m_i^j)
            // P_i^j[tx][y] = exp(S_i^j[tx][y] - m_i^j)
            float row_l = 0;
            for (int y = 0; y < Bc/block_width; y++) {
                //if (j * Bc + y >= N)
                //    break;  // break if we are done with the sequence
                //if (i * Br + tx < j * Bc + y)
                //    break;
                int col_offset = y * block_width;
                float exp_val = __expf(S[(Bc * row) + col + col_offset] - new_row_m);
                S[(Bc * row) + col + col_offset] = exp_val;
                // Sum over P_i^j to get row_sum(P_i^j)
                row_l += warp_reduce_sum_f32<WARP_SIZE/4>(exp_val);
            }

            // l_i^j = (exp(m_i^j-1 - m_i^j) * l_i^j-1) + row_sum(P_i^j)
            float row_m_exp = __expf(row_m_prev - new_row_m);
            float new_row_l = (row_m_exp * row_l_prev) + row_l;

            // O_i^j = diag(exp(m_i^j-1 - m_i^j))^-1 * O_i^j-1 + P_i^jVj
            for (int y = 0; y < d/block_width; y++) {
                float pv = 0;  // Pij * Vj
                int col_offset = y * block_width;
                for (int x = 0; x < Bc; x++) {
                    pv += S[(Bc * row) + x] *KVj[(x * d) + col + col_offset];
                }

                O[qkv_offset + (q_tile_size * i) + (row * d) + col + col_offset] = \
                    row_m_exp * O[qkv_offset + (q_tile_size * i) + (row * d) + col + col_offset] + pv;
            }

            // Update m and l
            row_m_prev = new_row_m;
            row_l_prev = new_row_l;
        }

        // O_i = diag(l_i^{Tc})^-1 * O_i^{Tc}
        for (int y = 0; y < d/block_width; y++) {
            //if (i * Br + tx < y * Bc)
            //    break;
            int col_offset = y * block_width;
            O[qkv_offset + (q_tile_size * i) + (row * d) + col + col_offset] /= row_l_prev;
        }
        // L_i = m_i^{Tc} + log(l_i^{Tc})
        // L[lm_offset + (Br * i) + tx] = row_m_prev + __logf(row_l_prev);
    }
}


torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 64; const int Br = 128;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, L to HBM
    auto O = torch::zeros_like(Q);
    // auto L = torch::zeros({B, nh, N});
    torch::Device device(torch::kCUDA);
    // L = L.to(device);
    O = O.to(device);

    // Calculate SRAM size needed per block
    int col_tile_size = Bc * d;  // size of Kj, Vj
    int row_tile_size = Br * d;  // size of Qi
    const int sram_size =
          (col_tile_size * sizeof(float))  // SRAM size for Kj, Vj
        + (row_tile_size * sizeof(float))  // SRAM size for Qi
        + (Bc * Br * sizeof(float));  // SRAM size for S

    //cudaFuncSetAttribute(flash_attention_2_forward_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(flash_attention_2_forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sram_size);

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(1024);  // Br x Br (1024) threads per block

    flash_attention_2_forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        O.data_ptr<float>()
    );
    return O;
}