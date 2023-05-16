import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union


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
