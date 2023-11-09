import numpy as np
from typing import Union, Tuple
from classify_fault.utils.check_run_time import elapsed

def detect_dynamics(data, dynamic_threshold, n=5):
    """
    Check for dynamic states in the data based on the specified threshold.
    
    Args:
        data (numpy.ndarray): 안정상태를 검사할 이전 데이터의 1D 배열.
        dynamic_threshold (float): 동적 임계값
        n (int): 관찰할 이전 데이터의 수

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
    if isinstance(data, np.ndarray):
        data = data.ravel()

    diff = np.abs(np.diff(data[-n:]))    # Calculate the absolute difference between consecutive data points
    avg_diff = np.mean(diff)

    if avg_diff >= dynamic_threshold:
        return True, avg_diff
    return False, avg_diff

def detect_dynamics_vec(data: np.ndarray,
                        dynamic_threshold: Union[float, np.ndarray],
                        n: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check for dynamic states in the data based on the specified threshold.

    Args:
        data (numpy.ndarray): 정상 상태를 검사할 이전 데이터의 2D 배열. 각 열은 변수를 나타냄.
        dynamic_threshold (float or np.ndarray): 동적 임계값. 단 값 또는 각 변수에 대한 임계값을 나타내는 1D 배열.
        n (int): 관찰할 이전 데이터의 수

    Returns:
        is_dynamic (np.ndarray): 동적 상태가 각 변수에서 검출되었는지 여부를 나타내는 부울.
        avg_diff (np.ndarray): 각 변수의 평균 차이를 나타내는 배열.

    Raises:
        ValueError: If data is not 2D, dynamic_threshold is not 1D or a single value, or n is not positive.

    Examples:
        data = np.array([[1, 2, 4, 6, 8, 10, 12],
                         [2, 4, 6, 8, 10, 12, 14]]).T    # 변수의 개수: 2, 데이터의 개수 7
        dynamic_threshold = np.array([2.5, 3.5])

        dynamic_detected, avg_diff = detect_dynamics_vec(data, dynamic_threshold)
        print(f"Is dynamic state detected? {dynamic_detected}")
        print(f"Average difference: {avg_diff}")
    """
    if data.ndim != 2:
        raise ValueError("The input 'data' must be a 2D array.")
    
    if not (np.isscalar(dynamic_threshold) or (isinstance(dynamic_threshold, np.ndarray) and dynamic_threshold.ndim == 1)):
        raise ValueError("The input 'dynamic_threshold' must be a single value or a 1D array.")

    if n <= 0:
        raise ValueError("The input 'n' must be a positive integer.")

    diff = np.abs(np.diff(data[-n:, :], axis=0))  # Calculate the absolute difference between consecutive data points
    avg_diff = np.mean(diff, axis=0)

    is_dynamic = avg_diff >= dynamic_threshold

    return is_dynamic, avg_diff
