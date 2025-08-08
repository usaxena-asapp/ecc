import torch
import numpy as np
import argparse
import time
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
from torch.utils.cpp_extension import load_inline

# ----------------- CUDA Kernel Source -----------------
cuda_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define IMAD(a, b, c) ((a) * (b) + (c))
#define iDivUp(a, b) ((a + b - 1) / b)

// Binary search function for finding threshold bin
__device__ inline int binary_search(const float* arr, int size, float value) {
    int left = 0;
    int right = size - 1;

    while (left < right) {
        int mid = (left + right) / 2;
        if (arr[mid] < value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__global__ void ECC_kernel_v30(
    const float* __restrict__ data,
    int* __restrict__ VCEC_device,
    const float* __restrict__ thresholds,
    int imageW,
    int imageH,
    int binNum
) {
    extern __shared__ float shmem[];

    // Warp-private histograms: warps_per_block * binNum
    const int warp_id = threadIdx.x / warpSize + threadIdx.y * (blockDim.x / warpSize);
    int warps_per_block = (blockDim.x * blockDim.y) / warpSize;float* warp_hist = shmem + warp_id * binNum;

    // Shared tile for image patch (+ halo)
    float* tile = shmem + (blockDim.x * blockDim.y / warpSize) * binNum;
    int tile_pitch = blockDim.x + 2;
    int tile_height = blockDim.y + 2;

    // Global coords
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    // Load tile with halo
    for (int dy = threadIdx.y; dy < tile_height; dy += blockDim.y) {
        for (int dx = threadIdx.x; dx < tile_pitch; dx += blockDim.x) {
            int imgx = blockIdx.x * blockDim.x + dx - 1;
            int imgy = blockIdx.y * blockDim.y + dy - 1;
            float val = 1e30f;
            if (imgx >= 0 && imgx < imageW && imgy >= 0 && imgy < imageH) {
                val = __ldg(&data[imgy * imageW + imgx]);
            }
            tile[dy * tile_pitch + dx] = val;
        }
    }
    __syncthreads();

    // Only process valid pixels
    if (gx < imageW && gy < imageH) {
        int lx = threadIdx.x + 1;
        int ly = threadIdx.y + 1;
        float c  = tile[ ly      * tile_pitch + lx     ];
        float t  = tile[(ly - 1) * tile_pitch + lx     ];
        float b  = tile[(ly + 1) * tile_pitch + lx     ];
        float l  = tile[ ly      * tile_pitch + lx - 1 ];
        float r  = tile[ ly      * tile_pitch + lx + 1 ];
        float tl = tile[(ly - 1) * tile_pitch + lx - 1 ];
        float tr = tile[(ly - 1) * tile_pitch + lx + 1 ];
        float bl = tile[(ly + 1) * tile_pitch + lx - 1 ];
        float br = tile[(ly + 1) * tile_pitch + lx + 1 ];

        // Compute change (branchless)
        float change = 1.0f;
        int c_l_l = (c < l);
        int c_lq_r = (c <= r);

        change -= (c < t);
        change += (c < t && c_l_l && c < tl);
        change += (c < t && c_lq_r && c < tr);

        change -= (c <= b);
        change += (c <= b && c_l_l && c <= bl);
        change += (c <= b && c_lq_r && c <= br);

        change -= c_l_l;
        change -= c_lq_r;

        // Binary search
        int bin = binary_search(thresholds, binNum, c);
        if (bin >= 0 && bin < binNum)
            atomicAdd(&warp_hist[warp_id * binNum + bin], change);
    }
    __syncthreads();

    // Reduce warp histograms into global histogram
    
    for (int bin = threadIdx.y * blockDim.x + threadIdx.x; bin < binNum; bin += blockDim.x * blockDim.y) {
        float sum = 0.0f;
        for (int w = 0; w < warps_per_block; w++) {
            sum += shmem[w * binNum + bin];
        }
        if (sum != 0.0f)
            atomicAdd(&VCEC_device[bin], __float2int_rn(sum));
    }
}

// Wrapper function matching the original computeECC signature
torch::Tensor computeECC(
    torch::Tensor data,
    torch::Tensor unique_values,
    int imageH,
    int imageW
) {
    int binNum = unique_values.size(0);
    auto VCEC_device = torch::zeros({binNum}, torch::dtype(torch::kInt32).device(data.device()));

    // Thread configuration logic from original
    int imageH_rounded = 512;
    if (binNum < 512) {
        imageH_rounded = 1;
        while (imageH_rounded < binNum) imageH_rounded *= 2;
        if (imageH_rounded > 512) imageH_rounded = 512;
    }

    dim3 threads;
    switch (imageH_rounded) {
        case 1: threads = dim3(512, 1); break;
        case 2: threads = dim3(256, 2); break;
        case 4: threads = dim3(128, 4); break;
        case 8: threads = dim3(64, 8); break;
        case 16: threads = dim3(32, 16); break;
        case 32: threads = dim3(16, 32); break;
        case 64: threads = dim3(16, 32); break;
        case 128: threads = dim3(16, 32); break;
        case 256: threads = dim3(16, 32); break;
        default: threads = dim3(16, 32);
    }

    if (binNum >= 1024) {
        printf("Warning: Too much shared memory used (%d bins), may fail\\n", binNum);
    }

    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));
    
    int warps_per_block = (threads.x * threads.y + 31) / 32;
    int tile_pitch = threads.x + 2;
    int tile_height = threads.y + 2;
    int tile_size = tile_pitch * tile_height;
    size_t shmem_size = sizeof(float) * (warps_per_block * binNum + tile_size);

    ECC_kernel_v30<<<blocks, threads, shmem_size>>>(
        data.data_ptr<float>(),
        VCEC_device.data_ptr<int>(),
        unique_values.data_ptr<float>(),
        imageW,
        imageH,
        binNum
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    return VCEC_device;
}
'''

cpp_source = '''
#include <torch/extension.h>

// Forward declaration
torch::Tensor computeECC(torch::Tensor data, torch::Tensor unique_values, int imageH, int imageW);

// Python binding
torch::Tensor ecc_v30(torch::Tensor data, torch::Tensor unique_values, int H, int W) {
    TORCH_CHECK(data.is_cuda(), "data must be a CUDA tensor");
    TORCH_CHECK(unique_values.is_cuda(), "unique_values must be a CUDA tensor");
    TORCH_CHECK(data.dtype() == torch::kFloat32, "data must be float32");
    TORCH_CHECK(unique_values.dtype() == torch::kFloat32, "unique_values must be float32");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    TORCH_CHECK(unique_values.is_contiguous(), "unique_values must be contiguous");

    return computeECC(data, unique_values, H, W);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ecc_v30", &ecc_v30, "ECC 2D v30 kernel (exact implementation)");
}
'''


# ----------------- File I/O -----------------
def load_dat(path, H, W, dtype=torch.float32, device='cuda'):
    """Load .dat file and convert to PyTorch tensor"""
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != H * W:
        raise ValueError(f"Expected {H * W} values, got {arr.size}")
    return torch.from_numpy(arr.reshape(H, W)).to(dtype=dtype, device=device)


def get_unique_values_sorted(data):
    """Extract unique values from data and sort them (ascending)"""
    unique_vals = torch.unique(data.flatten())
    unique_vals_sorted = torch.sort(unique_vals)[0]
    return unique_vals_sorted


# ----------------- Main -----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exact ECC v30 Implementation')
    parser.add_argument('-i', '--input', required=True, help='Input .dat file path')
    parser.add_argument('-H', '--height', type=int, required=True, help='Image height')
    parser.add_argument('-W', '--width', type=int, required=True, help='Image width')
    parser.add_argument('--output', type=str, default='ecc_v30', help='Output prefix for files')
    parser.add_argument('--max-bins', type=int, default=1024, help='Maximum number of bins')
    args = parser.parse_args()

    print("=" * 80)
    print("ECC v30 Kernel - Exact Implementation")
    print("=" * 80)

    # Compile CUDA extension
    print("\n📦 Compiling CUDA extension...")
    ecc_cuda = load_inline(
        name='ecc_v30_exact_opt',
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
        verbose=False
    )
    print("✅ CUDA extension compiled successfully")

    # Load data
    print(f"\n📂 Loading data from {args.input}...")
    t0 = time.time()
    data = load_dat(args.input, args.height, args.width)
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1e3

    print(f"✅ Data loaded: {args.height}x{args.width}")
    print(f"⏱️  Load time: {dt:.2f} ms")

    # Prepare data
    H, W = data.shape
    M = H * W
    data_flat = data.reshape(-1).contiguous()

    # Extract unique values (this is what the original does!)
    print(f"\n🔍 Extracting unique values from data...")
    unique_values = get_unique_values_sorted(data)
    binNum = len(unique_values)

    # Limit bins if necessary
    if binNum > args.max_bins:
        print(f"⚠️  Too many unique values ({binNum}), sampling {args.max_bins} evenly spaced values")
        indices = torch.linspace(0, binNum - 1, args.max_bins, dtype=torch.long)
        unique_values = unique_values[indices]
        binNum = args.max_bins

    print(f"📊 Number of unique values (bins): {binNum}")
    print(f"📊 Value range: [{unique_values[0].item():.3f}, {unique_values[-1].item():.3f}]")

    # Warm-up
    print("\n🔥 Warming up kernel...")
    for _ in range(3):
        _ = ecc_cuda.ecc_v30(data_flat, unique_values, H, W)
    torch.cuda.synchronize()

    # Benchmark
    print("\n📈 Running ECC_kernel_v30...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    num_iterations = 10
    start.record()
    for _ in range(num_iterations):
        hist_v30 = ecc_cuda.ecc_v30(data_flat, unique_values, H, W)
    end.record()
    torch.cuda.synchronize()

    t_v30 = start.elapsed_time(end) / num_iterations
    voxel_throughput = (H * W * binNum) / (t_v30 / 1000) / 1e9

    print(f"⏱️  Kernel time: {t_v30:.5f} ms (avg of {num_iterations} runs)")
    print(f"📊 Voxel throughput: {voxel_throughput:.2f} GVox/s")
    print(f"📊 Processing: {M:,} pixels × {binNum} bins = {M * binNum:,} operations")

    # Save results
    print(f"\n💾 Saving results...")
    np.savetxt(f"{args.output}_histogram.txt", hist_v30.cpu().numpy(), fmt='%d')
    np.savetxt(f"{args.output}_unique_values.txt", unique_values.cpu().numpy(), fmt='%.6f')

    # Create combined output
    combined = np.column_stack([
        unique_values.cpu().numpy(),
        hist_v30.cpu().numpy()
    ])
    np.savetxt(f"{args.output}_combined.csv", combined,
               delimiter=',', header='unique_value,ecc_count',
               comments='', fmt=['%.6f', '%d'])

    print(f"   Saved: {args.output}_histogram.txt")
    print(f"   Saved: {args.output}_unique_values.txt")
    print(f"   Saved: {args.output}_combined.csv")

    # Statistics
    print(f"\n📊 ECC Statistics:")
    print(f"   Total ECC sum: {hist_v30.sum().item()}")
    print(f"   Non-zero bins: {(hist_v30 != 0).sum().item()}/{binNum}")
    print(f"   Max count: {hist_v30.max().item()} at value {unique_values[hist_v30.argmax()].item():.3f}")
    print(f"   Min count: {hist_v30.min().item()}")

    # Sample output
    print(f"\n📊 First 10 histogram values:")
    for i in range(min(10, binNum)):
        print(f"   Value {unique_values[i].item():8.3f}: {hist_v30[i].item():6d}")

    print("\n✅ Processing complete!")