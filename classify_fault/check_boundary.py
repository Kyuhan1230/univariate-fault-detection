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
        # {'success': False, 'result': [True, 9.0]}
        # x 값이 하한 경계값보다 작으므로 하한을 초과합니다.
    """
    if (np.isnan(low) and np.isnan(high)) or (low is None and high is None):
        return {"success": False, "result": "No Boundary"}

    if low is None or np.isnan(low):
        clipped = np.clip(x, low, high)
        return {"success": x <= high, "result": [x > high, high] if x > high else [False, clipped]}

    if high is None or np.isnan(high):
        clipped = np.clip(x, low, high)
        return {"success": x >= low, "result": [x < low, low] if x < low else [False, clipped]}

    clipped = np.clip(x, low, high)
    return {"success": True, "result": [x < low or x > high, low if x < low else high] if clipped != x else [False, x]}


def set_boundary(statistics, boundary_type, x, sigma_level, tag_name):
    """
    주어진 통계 정보와 경계 유형, 입력값, 시그마 레벨, 태그 이름을 사용하여 경계 값을 계산합니다.
    
    Args:
        statistics (dict): 통계 정보를 저장하는 딕셔너리
        boundary_type (str): 경계 유형을 뜻하는 문자열
        x (float): 현재 데이터
        sigma_level (float): 시그마 레벨
        tag_name (str): 통계 정보에서 값을 가져올 태그 이름
    
    Returns:
        dict: 경계 값을 담은 딕셔너리
            - success (bool): 함수 실행 결과를 나타내는 값입니다. 경계 값을 계산할 수 있으면 True를 반환
            - result (list): 경계 값을 담은 리스트입니다. 계산에 실패하면 None을 반환
    """
    # 1. statistics 인자가 딕셔너리 타입인지 확인
    if not isinstance(statistics, dict):
        raise TypeError("The 'statistics' argument must be a dictionary.")
        
    # 2. boundary_type 인자가 문자열 타입인지 확인
    if not isinstance(boundary_type, str):
        raise TypeError("The 'boundary_type' argument must be a string.")
    
    valid_boundary_types = ["moving", "fix"]
    boundary_type = boundary_type.lower().replace(" ", "_")

    if boundary_type not in valid_boundary_types:
        raise ValueError(f"Invalid boundary type: {boundary_type}. Valid values should has {valid_boundary_types}.")
    
    # 3. x 인자가 실수 타입인지 확인
    if not isinstance(x, float):
        raise TypeError("The 'x' argument must be a float.")
        
    # 4. sigma_level 인자가 실수 타입인지 확인
    if not isinstance(sigma_level, float):
        raise TypeError("The 'sigma_level' argument must be a float.")

    # 5. tag_name 인자가 문자열 타입인지 확인
    if not isinstance(tag_name, str):
        raise TypeError("The 'tag_name' argument must be a string.")
    
    # 6. 경계 값 계산
    if "moving" in boundary_type:
        try:
            from classify_fault.update_avg_std import update_avg_std
        except ImportError:
            raise ImportError("The 'update_avg_std' function is required but could not be imported.")
        
        avg_old = statistics[tag_name]['avg']
        std_old = statistics[tag_name]['std']
        update_option = statistics[tag_name]['update_option']
        oldest_value = statistics[tag_name]['oldset_value']
        data_size = statistics[tag_name]['data_size']

        avg_updated, std_updated = update_avg_std(avg_old, std_old, x, update_option, oldest_value, data_size)
        statistics[tag_name]['avg'] = avg_updated
        statistics[tag_name]['std'] = std_updated
        
        warnings.warn("The oldest_value in schema has been updated.", Warning)
        
        high = avg_updated + sigma_level * std_updated
        low = avg_updated - sigma_level * std_updated
        return {"success": True, "result": [high, low]}

    else:
        avg = statistics[tag_name]['avg']
        std = statistics[tag_name]['std']
        high = avg + sigma_level * std
        low = avg - sigma_level * std
        return {"success": True, "result": [high, low]}


"""
네번째 함수)
schema의 tag_name별 oldest_value를 최신화하는 작업이다.
미정
"""