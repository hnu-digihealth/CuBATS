# config.py


gpu_acceleration = False


def set_gpu_acceleration(value):
    global gpu_acceleration
    gpu_acceleration = value


def get_gpu_acceleration():
    return gpu_acceleration


def has_nvidia_gpu():
    """
    Checks if the system has an NVIDIA GPU compatible with CuPy.

    Returns:
        bool: True if an NVIDIA GPU is available, False otherwise.
    """
    try:
        # Third Party
        import cupy as cp

        num_gpus = cp.cuda.runtime.getDeviceCount()
        return num_gpus > 0
    except ImportError:
        return False
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def get_array_module():
    """
    Returns the appropriate array module (NumPy or CuPy) based on the gpu_acceleration flag.

    Returns:
        module: The array module, either NumPy or CuPy.

    Raises:
        ValueError: If the array module is not set.
    """
    if gpu_acceleration:
        try:
            # Third Party
            import cupy as cp

            print("Using CuPy for GPU acceleration.")
            return cp
        except ImportError:
            print("CuPy is not available. Falling back to NumPy.")
    # Third Party
    import numpy as np

    print("Using NumPy.")
    return np
