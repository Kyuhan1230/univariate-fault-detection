import numpy as np
from typing import List, Optional, Dict, Union
from classify_fault.utils.check_run_time import elapsed


def detect_out_of_bounds(x: float, high: Optional[float] = None, low: Optional[float] = None) -> Dict[str, Union[bool, List[Union[bool, float]]]]:
    """
    현재 값이 경계를 초과하는지 검사합니다.
    Args:
        x (float): 현재 값을 나타내는 실수
        high (float, optional): 상한 경계값으로 None이나 nan이면 검사하지 않음.
        low (float, optional): 하한 경계값으로 None이나 nan이면 검사하지 않음.
    Returns:
        dict: 경계를 초과하는지 여부와 초과한 경계값을 담은 딕셔너리
            - success (bool): 함수 실행 결과를 나타내는 값
                경계를 초과하지 않으면 True를 반환
            - result (list): 경계를 초과하는지 여부와 초과한 경계값을 담은 리스트
                초과하지 않으면 [False, x]를 반환
    Examples:
        x = 10.5
        high = 10.0
        low = 9.0
        result = detect_out_of_bounds(x, high, low)
        print(result)
        # {'success': True, 'result': [False, 10.5]}
        # x 값이 하한과 상한 경계값 사이에 있으므로 경계를 초과하지 않습니다.
        high = None
        result = detect_out_of_bounds(x, high, low)
        print(result)
        # {'success': True, 'result': [True, 9.0]}
        # x 값이 하한 경계값보다 작으므로 하한을 초과합니다.
    """
    if (np.isnan(low) and np.isnan(high)) or (low is None and high is None):
        return {"success": False, "result": [None, None]}

    if low is None or np.isnan(low):
        clipped = np.clip(x, low, high)
        return {"success": True, "result": [x > high, high] if x > high else [False, clipped]}

    if high is None or np.isnan(high):
        clipped = np.clip(x, low, high)
        return {"success": True, "result": [x < low, low] if x < low else [False, clipped]}

    clipped = np.clip(x, low, high)
    return {"success": True, "result": [x < low or x > high, low if x < low else high] if clipped != x else [False, x]}


def detect_out_of_bounds_vec(x: np.ndarray,
                             high: Optional[np.ndarray] = None,
                             low: Optional[np.ndarray] = None) -> Dict[str, Union[bool, List[Union[bool, float]]]]:
    """
    현재 값이 경계를 초과하는지 검사합니다.
    Args:
        x (np.ndarray): 현재 값을 나내는 Numpy 배열(x1, ... , xN; N>=1)
        high (np.ndarray, optional): 상한 경계값으로 None이면 검사하지 않음.(high_x1, ... , high_xN; N>=1)
        low (np.ndarray, optional): 하한 경계값으로 None이면 검사하지 않음.(low_x1, ... , low_xN; N>=1)
    Returns:
        dict: 경계를 초과하는지 여부와 초과한 경계값을 담은 딕셔너리
            - success (np.ndarray(bool): 함수 실행 결과를 나타내는 값이 담긴 Numpy 배열
                배열 내에는 경계를 초과하지 않으면 True이 반환됨.
            - result (np.ndarray(list)): 경계를 초과하는 여부와 초과한 경계값을 담은 리스트
                초과하지 않으면 [False, x]를 반환
    Examples:
        x = np.array([10.5, 8.5])
        high = np.array([10.0, 9.0])
        low = np.array([9.0, 7.0])
        result = detect_out_of_bounds_vec(x, high, low)
        print(result)
        # {'success': np.array([True, True]), 'result': [np.array([False, False]), np.array([10.5, 8.5])]}
    """
    if high is None:
        high = np.full_like(x, np.inf)
    else:
        # None 값을 np.nan으로 변경하고, 그 다음에 np.nan을 np.INF로 변경
        high = np.where(high == None, np.nan, high)
        high = np.where(np.isnan(high), np.inf, high)

    if low is None:
        low = np.full_like(x, -np.inf)
    else:
        # None 값을 np.nan으로 변경하고, 그 다음에 np.nan을 np.NINF로 변경
        low = np.where(low == None, np.nan, low)
        low = np.where(np.isnan(low), -np.inf, low)

    success = np.logical_and(x >= low, x <= high)
    result = np.where(x < low, low, np.where(x > high, high, x))

    return {"success": success, "result": [np.logical_not(success), result]}
