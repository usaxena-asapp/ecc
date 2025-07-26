import numpy as np
import struct
import os
from scipy.ndimage import gaussian_filter
from scipy.fft import fftn, ifftn
import gc

# =============================================================================
# REALISTIC IMAGING DOMAIN CONFIGURATIONS
# =============================================================================

IMAGING_DOMAINS = {
    'medical_ct': {
        'dimensions': [(512, 512, 300), (512, 512, 500), (1024, 1024, 400)],
        'smoothness_xy': 1.5,  # Moderate smoothing in-plane
        'smoothness_z': 0.8,  # Less smoothing between slices
        'noise_level': 0.1,  # Low noise (good medical scanners)
        'value_range': (-1000, 3000),  # Hounsfield units
        'num_levels': 2048,  # High dynamic range
        'description': 'Medical CT scans with Hounsfield-like values'
    },

    'medical_mri': {
        'dimensions': [(256, 256, 100), (512, 512, 150), (256, 256, 200)],
        'smoothness_xy': 2.0,  # Smoother than CT
        'smoothness_z': 1.0,  # Moderate inter-slice correlation
        'noise_level': 0.15,  # Slightly more noise
        'value_range': (0, 4095),  # MRI intensity range
        'num_levels': 1024,
        'description': 'MRI scans with smooth tissue contrasts'
    },

    'micro_ct': {
        'dimensions': [(2048, 2048, 200), (2048, 2048, 500), (4096, 4096, 100)],
        'smoothness_xy': 0.8,  # Sharp features at high resolution
        'smoothness_z': 0.6,  # Sharp inter-slice transitions
        'noise_level': 0.05,  # Very low noise (high-quality scanners)
        'value_range': (0, 65535),  # 16-bit range
        'num_levels': 4096,  # High precision
        'description': 'High-resolution micro-CT with sharp material boundaries'
    },

    'confocal_microscopy': {
        'dimensions': [(1024, 1024, 50), (2048, 2048, 100), (1024, 1024, 200)],
        'smoothness_xy': 1.2,  # Biological structures
        'smoothness_z': 2.0,  # Thicker optical sections
        'noise_level': 0.2,  # Photon noise
        'value_range': (0, 4095),  # 12-bit typical
        'num_levels': 1024,
        'description': 'Confocal microscopy with fluorescence-like patterns'
    },

    'electron_microscopy': {
        'dimensions': [(4096, 4096, 20), (8192, 8192, 10), (4096, 4096, 50)],
        'smoothness_xy': 0.5,  # Very sharp ultrastructural details
        'smoothness_z': 0.3,  # Sharp section boundaries
        'noise_level': 0.08,  # Low noise but some grain
        'value_range': (0, 255),  # 8-bit grayscale
        'num_levels': 256,
        'description': 'Serial section electron microscopy with ultrastructural detail'
    },

    'satellite_hyperspectral': {
        'dimensions': [(8192, 8192, 5), (4096, 4096, 10), (16384, 16384, 3)],
        'smoothness_xy': 2.5,  # Large-scale geographic features
        'smoothness_z': 3.0,  # Correlated spectral bands
        'noise_level': 0.12,  # Atmospheric noise
        'value_range': (0, 2047),  # 11-bit typical
        'num_levels': 1024,
        'description': 'Satellite hyperspectral imagery with geographic features'
    },

    'astronomical': {
        'dimensions': [(4096, 4096, 3), (2048, 2048, 5), (8192, 8192, 4)],
        'smoothness_xy': 3.0,  # Cosmic structures
        'smoothness_z': 1.5,  # Different wavelength bands
        'noise_level': 0.25,  # Photon-limited
        'value_range': (0, 16383),  # 14-bit
        'num_levels': 2048,
        'description': 'Multi-wavelength astronomical observations'
    },

    'industrial_xray': {
        'dimensions': [(2048, 2048, 800), (1024, 1024, 1200), (4096, 4096, 200)],
        'smoothness_xy': 1.0,  # Material structures
        'smoothness_z': 0.9,  # Layer-like materials
        'noise_level': 0.08,  # Industrial quality
        'value_range': (0, 65535),  # 16-bit
        'num_levels': 2048,
        'description': 'Industrial X-ray tomography of materials'
    },

    'seismic_3d': {
        'dimensions': [(1024, 1024, 2000), (2048, 2048, 1000), (512, 512, 4000)],
        'smoothness_xy': 2.0,  # Geological layers
        'smoothness_z': 1.5,  # Stratigraphic continuity
        'noise_level': 0.15,  # Acquisition noise
        'value_range': (-32768, 32767),  # Signed 16-bit
        'num_levels': 1024,
        'description': 'Seismic reflection data with geological structures'
    }
}


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


def generate_grf_3d_realistic(height, width, depth, smoothness_xy=1.0, smoothness_z=1.0,
                              noise_level=0.1, seed=None):
    """
    Generate a realistic 3D GRF with different smoothness in XY vs Z directions.
    This better matches real imaging where Z-resolution is often different.
    """
    if seed is not None:
        np.random.seed(seed)

    total_size = height * width * depth

    # Use FFT approach for smaller volumes (< 256^3)
    if total_size < 256 ** 3:
        print(f"Small 3D volume ({depth}x{height}x{width}), using FFT approach...")

        # Generate white noise
        noise = np.random.randn(depth, height, width)

        # Apply anisotropic Gaussian smoothing in frequency domain
        kz = np.fft.fftfreq(depth, d=1.0)
        ky = np.fft.fftfreq(height, d=1.0)
        kx = np.fft.fftfreq(width, d=1.0)
        kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')

        # Anisotropic power spectrum
        k_squared_xy = kx_grid ** 2 + ky_grid ** 2
        k_squared_z = kz_grid ** 2
        power_spectrum = np.exp(-0.5 * (k_squared_xy * smoothness_xy ** 2 +
                                        k_squared_z * smoothness_z ** 2))

        # Apply filter in frequency domain
        noise_fft = fftn(noise)
        filtered_fft = noise_fft * np.sqrt(power_spectrum)
        grf = np.real(ifftn(filtered_fft))

        # Add additional noise if specified
        if noise_level > 0:
            additional_noise = np.random.randn(*grf.shape) * noise_level
            grf += additional_noise

        return grf

    else:
        # Use memory-efficient streaming approach for large volumes
        print(f"Large 3D volume ({depth}x{height}x{width}), using realistic streaming approach...")
        return generate_grf_3d_realistic_streaming(height, width, depth, smoothness_xy,
                                                   smoothness_z, noise_level, seed)


def generate_grf_3d_realistic_streaming(height, width, depth, smoothness_xy=1.0,
                                        smoothness_z=1.0, noise_level=0.1, seed=None):
    """
    Memory-efficient 3D GRF generation with realistic anisotropic properties.
    """
    if seed is not None:
        np.random.seed(seed)

    print(f"Generating realistic {depth}x{height}x{width} GRF...")

    # Initialize output array
    grf = np.zeros((depth, height, width), dtype=np.float32)

    # Z-direction correlation based on smoothness_z
    rho_z = np.exp(-1.0 / max(smoothness_z, 0.1))

    # Generate slice by slice
    prev_slice = None

    for z in range(depth):
        if z % max(1, depth // 20) == 0:
            print(f"  Processing slice {z}/{depth} ({100 * z / depth:.1f}%)")

        # Generate current slice
        if prev_slice is None:
            # First slice - pure random
            current_slice = np.random.randn(height, width).astype(np.float32)
        else:
            # Correlated with previous slice in Z direction
            noise = np.random.randn(height, width).astype(np.float32)
            current_slice = rho_z * prev_slice + np.sqrt(1 - rho_z ** 2) * noise

        # Apply XY-plane smoothing
        if smoothness_xy > 0:
            current_slice = gaussian_filter(current_slice, sigma=smoothness_xy, mode='reflect')

        # Add noise if specified
        if noise_level > 0:
            slice_noise = np.random.randn(height, width).astype(np.float32) * noise_level
            current_slice += slice_noise

        # Store current slice
        grf[z] = current_slice
        prev_slice = current_slice.copy()

        # Periodically clean up memory
        if z % 50 == 0:
            gc.collect()

    return grf


def generate_domain_specific_grf(domain_name, dimension_idx=0, seed=None):
    """
    Generate GRF data specific to a particular imaging domain.

    Args:
        domain_name: Key from IMAGING_DOMAINS
        dimension_idx: Which dimension set to use (0, 1, or 2)
        seed: Random seed

    Returns:
        GRF data with domain-appropriate characteristics
    """
    if domain_name not in IMAGING_DOMAINS:
        raise ValueError(f"Unknown domain: {domain_name}. Available: {list(IMAGING_DOMAINS.keys())}")

    domain = IMAGING_DOMAINS[domain_name]

    if dimension_idx >= len(domain['dimensions']):
        dimension_idx = 0
        print(f"Warning: dimension_idx too large, using 0")

    height, width, depth = domain['dimensions'][dimension_idx]

    print(f"Generating {domain_name} data:")
    print(f"  Dimensions: {depth}x{height}x{width}")
    print(f"  Description: {domain['description']}")

    # Generate the GRF with domain-specific parameters
    grf = generate_grf_3d_realistic(
        height=height,
        width=width,
        depth=depth,
        smoothness_xy=domain['smoothness_xy'],
        smoothness_z=domain['smoothness_z'],
        noise_level=domain['noise_level'],
        seed=seed
    )

    return grf, domain


def generate_domain_specific_chunked_save(domain_name, dimension_idx=0, seed=None, filename=None):
    """
    Generate domain-specific GRF directly to disk for large volumes.
    """
    if domain_name not in IMAGING_DOMAINS:
        raise ValueError(f"Unknown domain: {domain_name}")

    domain = IMAGING_DOMAINS[domain_name]
    if dimension_idx >= len(domain['dimensions']):
        dimension_idx = 0

    height, width, depth = domain['dimensions'][dimension_idx]

    if filename is None:
        filename = f"temp_{domain_name}_{depth}x{height}x{width}.dat"

    print(f"Generating {domain_name} data directly to disk:")
    print(f"  Dimensions: {depth}x{height}x{width}")
    print(f"  File: {filename}")

    if seed is not None:
        np.random.seed(seed)

    # Z-direction correlation
    rho_z = np.exp(-1.0 / max(domain['smoothness_z'], 0.1))

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

            # Apply XY smoothing
            if domain['smoothness_xy'] > 0:
                current_slice = gaussian_filter(current_slice, sigma=domain['smoothness_xy'], mode='reflect')

            # Add noise
            if domain['noise_level'] > 0:
                slice_noise = np.random.randn(height, width).astype(np.float32) * domain['noise_level']
                current_slice += slice_noise

            # Write slice to file
            current_slice_flat = current_slice.flatten(order='C')
            for value in current_slice_flat:
                f.write(struct.pack('f', value))

            prev_slice = current_slice
            del current_slice

            if z % 20 == 0:
                gc.collect()

    return filename, domain


def quantize_to_domain_levels(data, domain):
    """
    Quantize GRF to domain-specific value range and levels.
    """
    # Normalize to [0, 1]
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        normalized = np.zeros_like(data)
    else:
        normalized = (data - data_min) / (data_max - data_min)

    # Map to domain-specific range
    value_min, value_max = domain['value_range']
    mapped = normalized * (value_max - value_min) + value_min

    # Quantize to specified number of levels
    num_levels = domain['num_levels']
    if value_min < 0:  # Handle signed ranges
        quantized = np.round(mapped).astype(np.float32)
    else:
        # For unsigned ranges, use floor quantization
        normalized_to_levels = (mapped - value_min) / (value_max - value_min)
        quantized_levels = np.floor(normalized_to_levels * (num_levels - 1))
        quantized = (quantized_levels / (num_levels - 1)) * (value_max - value_min) + value_min
        quantized = quantized.astype(np.float32)

    return quantized


def quantize_file_to_domain(input_filename, output_filename, total_size, domain, chunk_size=1024 * 1024):
    """
    Quantize a large binary file to domain-specific ranges.
    """
    print(f"Quantizing to {domain['description']} range {domain['value_range']}...")

    # First pass: find min/max
    min_val = float('inf')
    max_val = float('-inf')

    with open(input_filename, 'rb') as f:
        processed = 0
        while processed < total_size:
            chunk_len = min(chunk_size, total_size - processed)
            chunk_bytes = f.read(chunk_len * 4)
            if not chunk_bytes:
                break

            chunk_data = np.frombuffer(chunk_bytes, dtype=np.float32)
            min_val = min(min_val, np.min(chunk_data))
            max_val = max(max_val, np.max(chunk_data))
            processed += len(chunk_data)

    # Second pass: quantize to domain range
    value_min, value_max = domain['value_range']
    num_levels = domain['num_levels']

    with open(input_filename, 'rb') as f_in, open(output_filename, 'wb') as f_out:
        processed = 0
        while processed < total_size:
            chunk_len = min(chunk_size, total_size - processed)
            chunk_bytes = f_in.read(chunk_len * 4)
            if not chunk_bytes:
                break

            chunk_data = np.frombuffer(chunk_bytes, dtype=np.float32)

            # Normalize and map to domain range
            if max_val != min_val:
                normalized = (chunk_data - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(chunk_data)

            mapped = normalized * (value_max - value_min) + value_min

            # Quantize
            if value_min < 0:
                quantized = np.round(mapped).astype(np.float32)
            else:
                norm_to_levels = (mapped - value_min) / (value_max - value_min)
                quant_levels = np.floor(norm_to_levels * (num_levels - 1))
                quantized = (quant_levels / (num_levels - 1)) * (value_max - value_min) + value_min
                quantized = quantized.astype(np.float32)

            for value in quantized:
                f_out.write(struct.pack('f', value))

            processed += len(chunk_data)


def save_as_binary(data, filename):
    """Save data as binary file compatible with the GPU ECC code."""
    data_flat = data.astype(np.float32).flatten(order='C')
    with open(filename, 'wb') as f:
        for value in data_flat:
            f.write(struct.pack('f', value))


def generate_realistic_dataset():
    """
    Generate a realistic dataset covering different imaging domains.
    """
    print("Generating Realistic Imaging Dataset")
    print("=====================================")

    for domain_name in IMAGING_DOMAINS.keys():
        print(f"\nGenerating {domain_name} samples...")
        domain = IMAGING_DOMAINS[domain_name]

        # Create domain directory
        domain_dir = f"RealisticImaging/{domain_name}"
        os.makedirs(domain_dir, exist_ok=True)

        # Generate 3 samples for each dimension
        for dim_idx, dimensions in enumerate(domain['dimensions']):
            height, width, depth = dimensions
            total_voxels = height * width * depth

            print(f"  Size {dim_idx}: {depth}x{height}x{width} ({total_voxels / 1e6:.1f}M voxels)")

            for sample_idx in range(3):
                filename = f"{domain_name}_{depth}x{height}x{width}_{sample_idx}.dat"
                filepath = f"{domain_dir}/{filename}"

                seed = hash(f"{domain_name}_{dim_idx}_{sample_idx}") % 2 ** 31

                # Choose generation method based on size
                if total_voxels < 100e6:  # < 100M voxels, load in memory
                    print(f"    Generating {filename} (in-memory)...")
                    grf, domain_info = generate_domain_specific_grf(domain_name, dim_idx, seed)
                    quantized = quantize_to_domain_levels(grf, domain_info)
                    save_as_binary(quantized, filepath)
                    del grf, quantized
                else:  # Large volumes, direct to disk
                    print(f"    Generating {filename} (direct-to-disk)...")
                    temp_file = f"temp_{filename}"
                    generate_domain_specific_chunked_save(domain_name, dim_idx, seed, temp_file)
                    quantize_file_to_domain(temp_file, filepath, total_voxels, domain)
                    os.remove(temp_file)

                # Print file info
                file_size_mb = os.path.getsize(filepath) / (1024 ** 2)
                print(f"      Saved: {filename} ({file_size_mb:.1f} MB)")

                gc.collect()


def generate_sample_from_domain(domain_name, dimension_idx=0, sample_idx=0):
    """
    Generate a single sample from a specific domain.
    """
    if domain_name not in IMAGING_DOMAINS:
        print(f"Available domains: {list(IMAGING_DOMAINS.keys())}")
        return None

    domain = IMAGING_DOMAINS[domain_name]
    height, width, depth = domain['dimensions'][dimension_idx]
    total_voxels = height * width * depth

    # Create output directory
    output_dir = f"RealisticImaging/{domain_name}"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{domain_name}_{depth}x{height}x{width}_{sample_idx}.dat"
    filepath = f"{output_dir}/{filename}"

    print(f"Generating {domain['description']}")
    print(f"Dimensions: {depth}x{height}x{width} ({total_voxels / 1e6:.1f}M voxels)")
    print(f"Output: {filepath}")

    seed = hash(f"{domain_name}_{dimension_idx}_{sample_idx}") % 2 ** 31

    # Generate based on size
    if total_voxels < 100e6:  # In-memory generation
        grf, domain_info = generate_domain_specific_grf(domain_name, dimension_idx, seed)
        quantized = quantize_to_domain_levels(grf, domain_info)
        save_as_binary(quantized, filepath)

        print(f"Generated:")
        print(f"  Shape: {quantized.shape}")
        print(f"  Value range: {np.min(quantized):.0f} to {np.max(quantized):.0f}")
        print(f"  File size: {os.path.getsize(filepath) / (1024 ** 2):.1f} MB")

        return quantized
    else:  # Direct-to-disk
        temp_file = f"temp_{filename}"
        generate_domain_specific_chunked_save(domain_name, dimension_idx, seed, temp_file)
        quantize_file_to_domain(temp_file, filepath, total_voxels, domain)
        os.remove(temp_file)

        print(f"Generated directly to disk:")
        print(f"  File size: {os.path.getsize(filepath) / (1024 ** 2):.1f} MB")
        print(f"  Value range: {domain['value_range']}")

        return None  # Don't return large arrays


if __name__ == "__main__":
    print("🔬 Realistic Domain-Specific GRF Generator")
    print("==========================================\n")

    # Show available domains
    print("Available imaging domains:")
    for domain_name, domain in IMAGING_DOMAINS.items():
        dims = domain['dimensions']
        print(f"  {domain_name}:")
        print(f"    {domain['description']}")
        for i, (h, w, d) in enumerate(dims):
            size_mb = h * w * d * 4 / (1024 ** 2)
            print(f"    Size {i}: {d}x{h}x{w} ({size_mb:.0f}MB)")
        print()

    # Generate examples from different domains
    print("Generating examples...")

    # Medical CT (moderate size)
    print("1. Medical CT scan...")
    ct_data = generate_sample_from_domain('medical_ct', dimension_idx=0, sample_idx=0)

    # Confocal microscopy (small, manageable)
    print("\n2. Confocal microscopy...")
    confocal_data = generate_sample_from_domain('confocal_microscopy', dimension_idx=0, sample_idx=0)

    # Micro-CT (large, realistic high-res)
    print("\n3. High-resolution micro-CT...")
    microct_data = generate_sample_from_domain('micro_ct', dimension_idx=0, sample_idx=0)

    print("\n🎯 Summary:")
    print("- Generated realistic imaging data for different domains")
    print("- Each domain has appropriate dimensions, noise, and smoothness")
    print("- Large volumes use memory-efficient direct-to-disk generation")
    print("- Data includes domain-specific value ranges and quantization")

    print(f"\n📁 Files saved in RealisticImaging/ directory")
    print(f"\n🚀 Test with GPU ECC:")
    print(f"  GPU_ECC.exe s RealisticImaging/medical_ct/medical_ct_300x512x512_0.dat output.txt 512 512 300")
    print(
        f"  GPU_ECC.exe s RealisticImaging/confocal_microscopy/confocal_microscopy_50x1024x1024_0.dat output.txt 1024 1024 50")

    # Uncomment to generate full realistic dataset
    # print("\n⚠️  Uncomment generate_realistic_dataset() to create full dataset")
    # generate_realistic_dataset()