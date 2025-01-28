def calculate_quantization_requirements(param_count, gpu_memory_mib, overhead=0.25):
    """
    Calculate the required quantization level to fit a model into GPU memory using MiB.
    
    Args:
        param_count (float): Number of model parameters (e.g., 13e9 for a 13B model).
        gpu_memory_mib (float): Total GPU memory in MiB (e.g., 32768 for a 32GB GPU).
        overhead (float): Fractional overhead for memory usage (default is 25%).
    
    Returns:
        str: Recommended precision level.
        float: Bytes per parameter.
    """
    # Effective memory in MiB after accounting for overhead
    effective_memory_mib = gpu_memory_mib / (1 + overhead)

    # Required bytes per parameter
    bytes_per_param = (effective_memory_mib * 1024**2) / param_count

    # Determine recommended precision level
    if bytes_per_param >= 4:
        precision = "FP32 (no quantization)"
    elif bytes_per_param >= 2:
        precision = "FP16 (half precision)"
    elif bytes_per_param >= 1:
        precision = "8-bit quantization"
    elif bytes_per_param >= 0.5:
        precision = "4-bit quantization"
    else:
        precision = "Less than 4-bit (experimental quantization required)"
    
    return precision, bytes_per_param

param_count = 70e9  # Number of parameters (e.g., 13B model) 13e9
gpu_memory_mib = 32768  # GPU memory in MiB (e.g., 32GB is 32768 MiB)

precision, bytes_per_param = calculate_quantization_requirements(param_count, gpu_memory_mib)
print(f"Recommended Precision: {precision}")
print(f"Bytes per Parameter: {bytes_per_param:.2f}")
