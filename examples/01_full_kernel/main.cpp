#include "../utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <hip/hip_runtime.h>

using namespace cute;

using TileShape = Shape<_128, _128, _16>;
using TiledMma = decltype(make_tiled_mma(CDNA2_32x32x2_F32F32F32F32_TN{},
                                         Layout<Shape<_2, _2, _1>>{}));

using SmemLayoutAtom = decltype(Layout<Shape<_32, Shape<_2, _4, _2>>,
                                       Stride<_16, Stride<_4, _1, _8>>>{});
using SmemLayoutAtomSwizzle =
    decltype(composition(Swizzle<2, 2, 3>{}, SmemLayoutAtom{}));
using SmemLayout =
    decltype(tile_to_shape(SmemLayoutAtomSwizzle{}, Shape<_128, _16, _2>{}));

using Tiledcopy = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<float>, float>{},
    Layout<Shape<_16, _16>, Stride<_16, _1>>{}, Layout<Shape<_1, _1>>{}));

// assume it is row-major
__global__ void __launch_bounds__(256)
    gemm_kernel(float *C, float const *A, float const *B, int M, int N, int K) {
  int blkx = blockIdx.x;
  int blky = blockIdx.y;
  int tid = threadIdx.x;

  Tensor mA = make_tensor(make_gmem_ptr(A),
                          make_layout(make_shape(M, K), GenRowMajor{}));
  Tensor mB = make_tensor(make_gmem_ptr(B),
                          make_layout(make_shape(K, N), GenRowMajor{}));

  // workgroup tiling
  TileShape blk_shape{};
  auto group_coord = make_coord(blkx, blky, _);
  Tensor gA = local_tile(mA, blk_shape, group_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, blk_shape, group_coord, Step<X, _1, _1>{});

  // shared memory
  __shared__ float smem_A[cosize(SmemLayout{})];
  __shared__ float smem_B[cosize(SmemLayout{})];
  Tensor sA = make_tensor(make_smem_ptr(smem_A), SmemLayout{});
  Tensor sB = make_tensor(make_smem_ptr(smem_B), SmemLayout{});

  // Tiled Copy
  Tiledcopy tiled_copy{};
  auto thr_copy = tiled_copy.get_slice(tid);

  Tensor tAgA = thr_copy.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = thr_copy.partition_D(sA); // (ACPY,ACPY_M,ACPY_K)
  Tensor tBgB = thr_copy.partition_S(gB); // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = thr_copy.partition_D(sB); // (BCPY,BCPY_N,BCPY_K)

  // Allocate the register tiles for double buffering -- same shape as
  // partitioned data
  Tensor tArA = make_fragment_like(tAsA(_, _, _, 0)); // (ACPY,ACPY_M,ACPY_K)
  Tensor tBrB = make_fragment_like(tBsB(_, _, _, 0)); // (BCPY,BCPY_N,BCPY_K)

  // Tile MMA compute thread partitions and allocate accumulators
  TiledMma tiled_mma{};
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0)); // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0)); // (MMA,MMA_M,MMA_K)

  Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_N,MMA_K,k)
  Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,k)

  Tensor accum = partition_fragment_C(
      tiled_mma, take<0, 2>(blk_shape)); // (MMA,MMA_N,MMA_M)

  // Copy gmem to rmem for the first k
  copy(tiled_copy, tAgA(_, _, _, 0), tArA);
  copy(tiled_copy, tBgB(_, _, _, 0), tBrB);
  // Copy rmem to smem
  copy(tArA, tAsA(_, _, _, 0));
  copy(tBrB, tBsB(_, _, _, 0));
  // Clear accumulators
  clear(accum);
  __syncthreads();

  // Load A, B smem->rmem for k=0
  copy_vec<uint_bit_t<128>>(tCsA(_, _, _, 0), tCrA);
  copy_vec<uint_bit_t<128>>(tCsB(_, _, _, 0), tCrB);

  //
  // Mainloop
  //

  // Size of the k-tiles's outer product mode (k)
  auto K_BLOCK_MAX = size<2>(tCrA);
  int k_tile_count = ceil_div(K, size<2>(blk_shape));
  __builtin_assume(k_tile_count >= 1);
  CUTLASS_PRAGMA_NO_UNROLL
  for (int k = 0; k < k_tile_count; ++k) {
    // Copy gmem to rmem for the next k
    if (k + 1 < k_tile_count) {
      copy(tiled_copy, tAgA(_, _, _, k + 1), tArA);
      copy(tiled_copy, tBgB(_, _, _, k + 1), tBrB);
    }
    cute::gemm(tiled_mma, accum, tCrA, tCrB, accum);
    // Copy rmem->smem for the next k
    if (k + 1 < k_tile_count) {
      copy(tCrA, tAsA(_, _, _, k + 1));
      copy(tCrB, tBsB(_, _, _, k + 1));
      __syncthreads();
      copy_vec<float4>(tCsA(_, _, _, k + 1), tCrA);
      copy_vec<float4>(tCsB(_, _, _, k + 1), tCrB);
    }
  }
  Tensor mC = make_tensor(make_gmem_ptr(C),
                          make_layout(make_shape(M, N), GenRowMajor{}));
  Tensor gC = local_tile(mC, blk_shape, group_coord, Step<_1, _1, X>{});
  auto tCsC = thr_mma.partition_C(gC);
  copy(accum, tCsC);
}

__global__ void reference_gemm(float *C, float const *A, float const *B, int M,
                               int N, int K) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < M && j < N) {
    float sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <M> <N> <K>\n", argv[0]);
    return 1;
  }
  int M = std::atoi(argv[1]);
  int N = std::atoi(argv[2]);
  int K = std::atoi(argv[3]);

  float *dA, *dB, *dC;
  HIP_CALL(hipMalloc(&dA, M * K * sizeof(float)));
  HIP_CALL(hipMalloc(&dB, K * N * sizeof(float)));
  HIP_CALL(hipMalloc(&dC, M * N * sizeof(float)));

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C(M * N);
  std::vector<float> C_ref(M * N);

  for (int i = 0; i < M * K; ++i) {
    A[i] = (i % 10) / 100;
    B[i] = ((i % 10) - 5) / 100;
  }

  HIP_CALL(
      hipMemcpy(dA, A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
  HIP_CALL(
      hipMemcpy(dB, B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(gemm_kernel, dim3(N / 128, M / 128), dim3(256), 0, 0, dC,
                     dA, dB, M, N, K);

  hipEvent_t start, stop;
  HIP_CALL(hipEventCreate(&start));
  HIP_CALL(hipEventCreate(&stop));
  HIP_CALL(hipEventRecord(start));
  for (int i = 0; i < 10; ++i) {
    hipLaunchKernelGGL(gemm_kernel, dim3(N / 128, M / 128), dim3(256), 0, 0, dC,
                       dA, dB, M, N, K);
  }
  HIP_CALL(hipEventRecord(stop));
  HIP_CALL(hipEventSynchronize(stop));
  float ms;
  HIP_CALL(hipEventElapsedTime(&ms, start, stop));
  ms /= 10;
  double flops = 2.0 * M * N * K / (ms / 1e3);

  printf("GFLOPS: %.2f\n", flops / 1e9);

  HIP_CALL(
      hipMemcpy(C.data(), dC, M * N * sizeof(float), hipMemcpyDeviceToHost));

  hipLaunchKernelGGL(reference_gemm, dim3(N / 15, M / 15), dim3(16, 16), 0, 0,
                     dC, dA, dB, M, N, K);
  HIP_CALL(hipMemcpy(C_ref.data(), dC, M * N * sizeof(float),
                     hipMemcpyDeviceToHost));

  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      if (std::abs(C[r * N + c] - C_ref[r * N + c]) > 1e-6) {
        printf("Mismatch at (%d, %d): %f != %f\n", r, c, C[r * N + c],
               C_ref[r * N + c]);
        return 1;
      }
    }
  }
  printf("Success!\n");
}