import numpy as np
import struct
import os
from scipy.ndimage import gaussian_filter
from scipy.fft import fftn, ifftn
import gc


def generate_grf_2d(height, width, smoothness=1.0, seed=None):
    """
    Generate a 2D Gaussian Random Field.
    Original FFT approach - works well for 2D.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate white noise
    noise = np.random.randn(height, width)

    # For large 2D images, use spatial domain filtering instead of FFT
    if height * width > 8192 * 8192:  # For very large 2D images
        print(f"Large 2D image ({height}x{width}), using spatial domain filtering...")
        if smoothness > 0:
            grf = gaussian_filter(noise.astype(np.float32), sigma=smoothness, mode='reflect')
        else:
            grf = noise.astype(np.float32)
        return grf

    # Apply Gaussian smoothing in frequency domain for better control
    # Create coordinate grids
    ky = np.fft.fftfreq(height, d=1.0)
    kx = np.fft.fftfreq(width, d=1.0)
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    # Power spectrum (Gaussian in frequency domain)
    k_squared = kx_grid ** 2 + ky_grid ** 2
    power_spectrum = np.exp(-0.5 * k_squared * smoothness ** 2)

    # Apply filter in frequency domain
    noise_fft = fftn(noise)
    filtered_fft = noise_fft * np.sqrt(power_spectrum)
    grf = np.real(ifftn(filtered_fft))

    return grf


def generate_grf_3d(height, width, depth, smoothness=1.0, seed=None):
    """
    Generate a 3D Gaussian Random Field.
    Uses memory-efficient approach for large volumes.
    """
    if seed is not None:
        np.random.seed(seed)

    total_size = height * width * depth

    # Use FFT approach for smaller volumes (< 512^3)
    if total_size < 512 ** 3:
        print(f"Small 3D volume ({depth}x{height}x{width}), using FFT approach...")
        # Generate white noise
        noise = np.random.randn(depth, height, width)  # Note: depth first for compatibility

        # Apply Gaussian smoothing in frequency domain
        kz = np.fft.fftfreq(depth, d=1.0)
        ky = np.fft.fftfreq(height, d=1.0)
        kx = np.fft.fftfreq(width, d=1.0)
        kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')

        # Power spectrum (Gaussian in frequency domain)
        k_squared = kx_grid ** 2 + ky_grid ** 2 + kz_grid ** 2
        power_spectrum = np.exp(-0.5 * k_squared * smoothness ** 2)

        # Apply filter in frequency domain
        noise_fft = fftn(noise)
        filtered_fft = noise_fft * np.sqrt(power_spectrum)
        grf = np.real(ifftn(filtered_fft))

        return grf

    else:
        # Use memory-efficient streaming approach for large volumes
        print(f"Large 3D volume ({depth}x{height}x{width}), using streaming approach...")
        return generate_grf_3d_streaming(height, width, depth, smoothness, seed)


def generate_grf_3d_streaming(height, width, depth, smoothness=1.0, seed=None):
    """
    Memory-efficient 3D GRF generation for large volumes.
    Generates slice-by-slice with correlation between slices.
    """
    if seed is not None:
        np.random.seed(seed)

    print(f"Generating {depth}x{height}x{width} GRF with streaming approach...")

    # Initialize output array
    grf = np.zeros((depth, height, width), dtype=np.float32)

    # Correlation parameter based on smoothness
    rho_z = np.exp(-1.0 / max(smoothness, 0.1))  # Z-direction correlation

    # Generate slice by slice
    prev_slice = None

    for z in range(depth):
        if z % max(1, depth // 20) == 0:  # Progress updates
            print(f"  Processing slice {z}/{depth} ({100 * z / depth:.1f}%)")

        # Generate current slice
        if prev_slice is None:
            # First slice - pure random
            current_slice = np.random.randn(height, width).astype(np.float32)
        else:
            # Correlated with previous slice in Z direction
            noise = np.random.randn(height, width).astype(np.float32)
            current_slice = rho_z * prev_slice + np.sqrt(1 - rho_z ** 2) * noise

        # Apply in-plane (XY) smoothing
        if smoothness > 0:
            sigma_xy = smoothness * 0.7  # Slightly less smoothing in XY plane
            current_slice = gaussian_filter(current_slice, sigma=sigma_xy, mode='reflect')

        # Store current slice
        grf[z] = current_slice
        prev_slice = current_slice.copy()

        # Periodically clean up memory
        if z % 50 == 0:
            gc.collect()

    return grf


def generate_grf_3d_chunked_save(height, width, depth, smoothness=1.0, seed=None, filename=None, chunk_size=64):
    """
    Alternative approach: Generate and save 3D GRF directly to disk in chunks.
    This avoids holding the entire volume in memory.
    Returns the filename instead of the data.
    """
    if seed is not None:
        np.random.seed(seed)

    if filename is None:
        filename = f"temp_grf_3d_{depth}x{height}x{width}.dat"

    print(f"Generating {depth}x{height}x{width} GRF directly to disk: {filename}")

    # Correlation parameter based on smoothness
    rho_z = np.exp(-1.0 / max(smoothness, 0.1))

    with open(filename, 'wb') as f:
        prev_slice = None

        for z in range(depth):
            if z % max(1, depth // 20) == 0:
                print(f"  Processing slice {z}/{depth} ({100 * z / depth:.1f}%)")

            # Generate current slice
            if prev_slice is None:
                current_slice = np.random.randn(height, width).astype(np.float32)
            else:
                noise = np.random.randn(height, width).astype(np.float32)
                current_slice = rho_z * prev_slice + np.sqrt(1 - rho_z ** 2) * noise

            # Apply in-plane smoothing
            if smoothness > 0:
                sigma_xy = smoothness * 0.7
                current_slice = gaussian_filter(current_slice, sigma=sigma_xy, mode='reflect')

            # Write slice to file
            current_slice_flat = current_slice.flatten(order='C')
            for value in current_slice_flat:
                f.write(struct.pack('f', value))

            prev_slice = current_slice

            # Clean up
            del current_slice
            if z % 20 == 0:
                gc.collect()

    print(f"Saved raw GRF to {filename}")
    return filename


def quantize_to_levels(data, num_levels=1024):
    """
    Quantize the GRF to a specific number of discrete levels.
    The paper mentions their GRF data contains 1024 unique values.
    """
    # Normalize to [0, 1]
    data_min = np.min(data)
    data_max = np.max(data)
    normalized = (data - data_min) / (data_max - data_min)

    # Quantize to num_levels
    quantized = np.floor(normalized * (num_levels - 1)).astype(np.int32)

    # Convert back to float32 for compatibility with the GPU code
    return quantized.astype(np.float32)


def quantize_file_streaming(input_filename, output_filename, total_size, num_levels=1024, chunk_size=1024 * 1024):
    """
    Quantize a large binary file in streaming fashion.
    For very large files that don't fit in memory.
    """
    print(f"Quantizing {input_filename} to {num_levels} levels...")

    # First pass: find min/max
    print("  First pass: finding min/max values...")
    min_val = float('inf')
    max_val = float('-inf')

    with open(input_filename, 'rb') as f:
        processed = 0
        while processed < total_size:
            chunk_len = min(chunk_size, total_size - processed)
            chunk_bytes = f.read(chunk_len * 4)  # 4 bytes per float
            if not chunk_bytes:
                break

            chunk_data = np.frombuffer(chunk_bytes, dtype=np.float32)
            min_val = min(min_val, np.min(chunk_data))
            max_val = max(max_val, np.max(chunk_data))

            processed += len(chunk_data)
            if processed % (10 * chunk_size) == 0:
                print(f"    Processed {processed}/{total_size} values")

    print(f"  Min: {min_val}, Max: {max_val}")

    # Second pass: quantize and write
    print("  Second pass: quantizing...")
    value_range = max_val - min_val
    if value_range == 0:
        value_range = 1

    with open(input_filename, 'rb') as f_in, open(output_filename, 'wb') as f_out:
        processed = 0
        while processed < total_size:
            chunk_len = min(chunk_size, total_size - processed)
            chunk_bytes = f_in.read(chunk_len * 4)
            if not chunk_bytes:
                break

            chunk_data = np.frombuffer(chunk_bytes, dtype=np.float32)
            normalized = (chunk_data - min_val) / value_range
            quantized = np.floor(normalized * (num_levels - 1)).astype(np.float32)

            for value in quantized:
                f_out.write(struct.pack('f', value))

            processed += len(chunk_data)
            if processed % (10 * chunk_size) == 0:
                print(f"    Processed {processed}/{total_size} values")


def save_as_binary(data, filename):
    """
    Save data as binary file compatible with the GPU ECC code.
    The code expects 32-bit IEEE 754 floating point values.
    """
    # Ensure data is float32 and in C-order (row-major)
    data_flat = data.astype(np.float32).flatten(order='C')

    with open(filename, 'wb') as f:
        for value in data_flat:
            f.write(struct.pack('f', value))


def generate_grf_dataset():
    """
    Generate a dataset similar to what was used in the GPU ECC paper.
    Based on the paper: 70 2D GRFs with 7 sizes (10 samples each)
    and 30 3D GRFs with 3 sizes (10 samples each).
    """

    # Create output directories
    os.makedirs("GaussRandomField/2D", exist_ok=True)
    os.makedirs("GaussRandomField/3D", exist_ok=True)

    # 2D GRF generation
    # 7 different sizes, 10 samples each
    sizes_2d = [64, 128, 256, 512, 1024, 2048, 4096]
    smoothness_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]  # Different smoothness levels

    print("Generating 2D Gaussian Random Fields...")
    for i, size in enumerate(sizes_2d):
        smoothness = smoothness_levels[i % len(smoothness_levels)]
        for sample_idx in range(10):
            print(f"  Generating 2D_{size}_{sample_idx}.dat")

            # Generate GRF
            grf = generate_grf_2d(size, size, smoothness=smoothness, seed=i * 10 + sample_idx)

            # Quantize to 1024 levels
            quantized_grf = quantize_to_levels(grf, num_levels=1024)

            # Save as binary file
            filename = f"GaussRandomField/2D/2D_{size}_{sample_idx}.dat"
            save_as_binary(quantized_grf, filename)

    # 3D GRF generation
    # 3 different sizes, 10 samples each
    sizes_3d = [64, 128, 256]

    print("Generating 3D Gaussian Random Fields...")
    for i, size in enumerate(sizes_3d):
        smoothness = smoothness_levels[i]
        for sample_idx in range(10):
            print(f"  Generating 3D_{size}_{sample_idx}.dat")

            # Generate GRF
            grf = generate_grf_3d(size, size, size, smoothness=smoothness, seed=100 + i * 10 + sample_idx)

            # Quantize to 1024 levels
            quantized_grf = quantize_to_levels(grf, num_levels=1024)

            # Save as binary file
            filename = f"GaussRandomField/3D/3D_{size}_{sample_idx}.dat"
            save_as_binary(quantized_grf, filename)


def generate_single_2d_grf_example(dim=128):
    """
    Generate a single example GRF for testing.
    """
    print(f"Generating single {dim}x{dim} 2D GRF example...")
    filename = f"2D_{dim}_gen.dat"

    # Generate a 2D GRF
    grf = generate_grf_2d(height=dim, width=dim, smoothness=1.5, seed=42)
    quantized_grf = quantize_to_levels(grf, num_levels=1024)

    # Create directory if needed
    os.makedirs(f"../../GaussRandomField/2D/{dim}", exist_ok=True)

    # Save the example
    save_as_binary(quantized_grf, f"../../GaussRandomField/2D/{dim}/{filename}")

    print(f"Generated example GRF:")
    print(f"  Shape: {quantized_grf.shape}")
    print(f"  Min value: {np.min(quantized_grf)}")
    print(f"  Max value: {np.max(quantized_grf)}")
    print(f"  Unique values: {len(np.unique(quantized_grf))}")
    print(f"  Saved as: {filename}")

    return quantized_grf


def generate_single_3d_grf_example(dim=128):
    """
    Generate a single example 3D GRF for testing.
    Automatically chooses the best approach based on size.
    """
    print(f"Generating single {dim}x{dim}x{dim} 3D GRF example...")
    filename = f"3D_{dim}_gen.dat"

    # Create directory if needed
    os.makedirs(f"../../GaussRandomField/3D/{dim}", exist_ok=True)
    output_path = f"../../GaussRandomField/3D/{dim}/{filename}"

    # For very large volumes, use direct-to-disk approach
    if dim >= 512:
        print(f"Large volume detected, using direct-to-disk generation...")

        # Generate directly to temporary file
        temp_filename = f"temp_3d_{dim}.dat"
        generate_grf_3d_chunked_save(dim, dim, dim, smoothness=1.5, seed=42, filename=temp_filename)

        # Quantize the file
        total_elements = dim * dim * dim
        quantize_file_streaming(temp_filename, output_path, total_elements, num_levels=1024)

        # Clean up temporary file
        os.remove(temp_filename)

        print(f"Generated example GRF:")
        print(f"  Shape: ({dim}, {dim}, {dim})")
        print(f"  Estimated unique values: ~1024")
        print(f"  Saved as: {filename}")
        print(f"  File size: {os.path.getsize(output_path) / (1024 ** 2):.1f} MB")

        return None  # Don't return the data for large volumes

    else:
        # For smaller volumes, use the normal approach
        grf = generate_grf_3d(height=dim, width=dim, depth=dim, smoothness=1.5, seed=42)
        quantized_grf = quantize_to_levels(grf, num_levels=1024)

        # Save the example
        save_as_binary(quantized_grf, output_path)

        print(f"Generated example GRF:")
        print(f"  Shape: {quantized_grf.shape}")
        print(f"  Min value: {np.min(quantized_grf)}")
        print(f"  Max value: {np.max(quantized_grf)}")
        print(f"  Unique values: {len(np.unique(quantized_grf))}")
        print(f"  Saved as: {filename}")

        return quantized_grf


if __name__ == "__main__":
    # Generate examples with different sizes to test both approaches
    print("=== Fast GRF Generator (Adapted) ===\n")

    # Small 2D example (fast)
    # print("1. Generating 2D example...")
    # single_2d_grf = generate_single_2d_grf_example(dim=1024)

    # Small 3D example (uses FFT)
    print("\n2. Generating small 3D example (will use FFT)...")
    small_3d_grf = generate_single_3d_grf_example(dim=512)

    # Large 3D example (uses streaming)
    # print("\n3. Generating large 3D example (will use streaming)...")
    # large_3d_grf = generate_single_3d_grf_example(dim=1024)

    # print("\n=== Summary ===")
    # print("- Small volumes (< 512³): Uses FFT approach (your original method)")
    # print("- Large volumes (≥ 512³): Uses streaming approach (memory efficient)")
    # print("- Very large volumes: Direct-to-disk generation")

    # print("\nTo use with GPU ECC code:")
    # print("  GPU_ECC.exe s ../../GaussRandomField/2D/512/2D_512_gen.dat output.txt 512 512 0")
    # print("  GPU_ECC.exe s ../../GaussRandomField/3D/128/3D_128_gen.dat output.txt 128 128 128")
    # print("  GPU_ECC.exe s ../../GaussRandomField/3D/1024/3D_1024_gen.dat output.txt 1024 1024 1024")