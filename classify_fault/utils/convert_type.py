import numpy as np

def numpy_to_python_type(value):
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    elif isinstance(value, (np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (float, int)):
        return value
    elif isinstance(value, np.ndarray) and value.ndim == 1:
        return float(value)
    else:
        raise TypeError(f"Unsupported data type: {type(value)}")