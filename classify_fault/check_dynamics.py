import numpy as np


def detect_dynamics(data, dynamic_threshold):
    """
    Check for dynamic states in the data based on the specified threshold.
    
    Args:
        data (numpy.ndarray): 안정상태를 검사할 이전 데이터의 1D 배열.
        dynamic_threshold (float): 동적 임계값
    
    Returns:
        bool: 동적 상태가 검출되었는지 여부.

    Examples:
        data = np.array([1, 2, 4, 6, 8, 10, 12])
        dynamic_threshold = 2.5

        dynamic_detected = detect_dynamics(data, dynamic_threshold)
        print(f"Is dynamic state detected? {dynamic_detected}")
        # Not in normal state (dynamic).
        # Is dynamic state detected? True
    """
    diff = np.abs(np.diff(data))    # Calculate the absolute difference between consecutive data points
    avg_diff = np.mean(diff)

    if avg_diff >= dynamic_threshold:
        return True
    return False
