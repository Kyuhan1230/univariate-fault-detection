import os
import yaml
import numpy as np
from .utils.convert_type import numpy_to_python_type


def calculate_cusum(data_point, average, cusum_threshold, C_plus=0, C_minus=0):
    """Tabular CUSUM 알고리즘을 사용하여 CUSUM을 계산합니다.
    Tabular CUSUM은 공정 변화를 감지하는데, 사용할 수 있습니다.

    Args:
        data_point (float): 모니터링할 데이터 포인트
        average (float): 모니터링 대상의 목표값(평균값)
        cusum_threshold (float): CUSUM 차트에서 경계값으로 사용할 임계값
        C_plus (float, optional): 양수의 누적값. Defaults to 0.
        C_minus (float, optional): 음수의 누적값. Defaults to 0.

    Raises:
        ValueError: average data_point이 float 형태가 아닌 경우
        ValueError: cusum_threshold가 양수가 아닌 경우

    Returns:
        C_plus (float): 양수의 누적값
        C_minus (float): 음수의 누적값
    
    Examples:
        >>> calculate_cusum(20, 25, 3)
        (0, 0)
        >>> calculate_cusum(22, 25, 3)
        (0, 0)
        >>> calculate_cusum(30, 25, 3)
        (5, 0)
        >>> calculate_cusum(20, 25, 3, C_plus=5, C_minus=0)
        (5, 0)
        >>> calculate_cusum(20, 25, 3, C_plus=5, C_minus=2)
        (5, 2)
    """
    # 입력값의 유효성 검사
    if not isinstance(average, (float, int)) or not isinstance(data_point, (float, int)):
        raise ValueError(f"average({type(average)})과 data_point{type(data_point)}은 float 혹은 integer 형태이어야 합니다.")
    
    if cusum_threshold <= 0:
        raise ValueError("cusum_threshold는 양수이어야 합니다.")
    
    # CUSUM 알고리즘 수행
    C_plus_ = max(0, C_plus + data_point - average - cusum_threshold)
    C_minus_ = min(0, C_minus - data_point + average - cusum_threshold)
    
    # CUSUM False 알람 최소화를 위한 추가 알고리즘
    C_plus = C_plus_ if C_plus_ >= C_plus else 0
    C_minus = C_minus_ if C_minus_ <= C_minus else 0
    print(round(C_plus, 3), round(C_minus, 3), round(data_point, 3), round(average, 3), 
          round(cusum_threshold, 3), round(C_minus + data_point - (average - cusum_threshold), 3))
    return C_plus, C_minus



def calculate_ewma(data_point, ewma_alpha, smoothed_data=None):
    """EWMA 알고리즘을 사용하여 입력값의 drift를 추적합니다.

    Args:
        data_point (float): 모니터링할 데이터 포인트
        ewma_alpha (float): EWMA 차트에서 smoothing factor로 사용할 알파값
        smoothed_data (float, optional): 이전 데이터 포인트의 smoothing된 값. Defaults to None.

    Raises:
        ValueError: data_point와 smoothed_data가 float 형태가 아닌 경우
        ValueError: ewma_alpha가 0보다 작거나 1보다 큰 경우

    Returns:
        smoothed_data (float): 현재 데이터 포인트의 smoothing된 값
    
    Examples:
        data_point = 8.2
        ewma_alpha = 0.1
        smoothed_data = 9.0

        smoothed_data = calculate_ewma(data_point, ewma_alpha, smoothed_data=smoothed_data)
        print(f"Smoothed data: {smoothed_data}")
    """
    # 입력값의 유효성 검사
    if not isinstance(data_point, (float, int)) or not isinstance(smoothed_data, ((float, int), type(None))):
        raise ValueError("data_point와 smoothed_data는 float 형태이어야 합니다.")

    if not 0 < ewma_alpha < 1:
        raise ValueError("ewma_alpha는 0과 1 사이의 값이어야 합니다.")
    
    # EWMA 알고리즘 수행
    if smoothed_data is None:
        smoothed_data = data_point
    else:
        smoothed_data = ewma_alpha * data_point + (1 - ewma_alpha) * smoothed_data
    
    return smoothed_data



def detect_drift(data_point: float, average: float, 
                 cusum_threshold=None, 
                 ewma_alpha=None, 
                 C_plus=0, C_minus=0, 
                 smoothed_data=None,
                 cusum_limit=10):
    """CUSUM과 EWMA 알고리즘을 사용하여 입력값의 drift를 추적하고 결과를 반환합니다.

    Args:
        data_point (float): 모니터링할 데이터 포인트
        average (float): 모니터링 대상의 목표값(평균값)
        cusum_threshold (float, optional): CUSUM 차트에서 경계값으로 사용할 임계값. Defaults to None.
        ewma_alpha (float, optional): EWMA 차트에서 smoothing factor로 사용할 알파값. Defaults to None.
        C_plus (float, optional): 양수의 누적값. Defaults to 0.
        C_minus (float, optional): 음수의 누적값. Defaults to 0.
        smoothed_data (float, optional): 이전 데이터 포인트의 smoothing된 값. Defaults to None.

    Returns:
        dict: 입력값의 drift 여부와 결과값
            {
                "CUSUM": {
                    "detected": bool, # drift가 감지되었는지 여부
                    "C_plus": float, # 양수의 누적값
                    "C_minus": float # 음수의 누적값
                },
                "EWMA": {
                    "detected": bool, # drift가 감지되었는지 여부
                    "smoothed_data": float # 현재 데이터 포인트의 smoothing된 값
                }
            }
    data_point = 15.0
    average = 10.0
    cusum_threshold = 4.0
    ewma_alpha = 0.5

    result = detect_drift(data_point, average, cusum_threshold=cusum_threshold, ewma_alpha=ewma_alpha)
    print(result)
    # {
    #     'CUSUM': {'detected': True, 'C_plus': 5.0, 'C_minus': 0},
    #     'EWMA': {'detected': True, 'smoothed_data': 12.5}
    # }
    # CUSUM과 EWMA 알고리즘 모두 drift가 감지되었습니다.

    data_point = 9.0

    result = detect_drift(data_point, average, cusum_threshold=cusum_threshold, ewma_alpha=ewma_alpha)
    print(result)
    # {
    #     'CUSUM': {'detected': False, 'C_plus': 0, 'C_minus': 0},
    #     'EWMA': {'detected': False, 'smoothed_data': 9.5}
    # }
    # CUSUM과 EWMA 알고리즘 모두 drift가 감지되지 않았습니다.

    """
    if cusum_threshold is None and ewma_alpha is None:
        raise ValueError("At least one of cusum_threshold and ewma_alpha must be provided.")
    
    result = {}

    if cusum_threshold is not None:
        # Check Type
        cusum_threshold = numpy_to_python_type(cusum_threshold)
        data_point = numpy_to_python_type(data_point)
        # Calculate CUSUM
        C_plus, C_minus = calculate_cusum(data_point, average, cusum_threshold, C_plus=C_plus, C_minus=C_minus)
        # Detected Drift
        detected = (C_plus > cusum_limit) or (C_minus < -cusum_limit)
        result['CUSUM'] = {'detected': detected, 'C_plus': C_plus, 'C_minus': C_minus}
    
    if ewma_alpha is not None:
        # Check Type
        ewma_alpha = numpy_to_python_type(ewma_alpha)
        data_point = numpy_to_python_type(data_point)
        if smoothed_data is None:
            smoothed_data = average
        smoothed_data = numpy_to_python_type(smoothed_data)
        # Calculate EWMA
        smoothed_data = calculate_ewma(data_point, ewma_alpha, smoothed_data=smoothed_data)
        # Detected Drift (After CUSUM)
        detected = abs(smoothed_data - average) > cusum_threshold if cusum_threshold is not None else False
        result['EWMA'] = {'detected': detected, 'smoothed_data': smoothed_data}

    return result

def update_drift_result(config_path: str, tag: str, drift_result: dict):
    if tag is None:
        return
    # 파일이 존재하는지 확인
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File '{config_path}' not found.")
        
    # tag 변수가 문자열인지 확인
    if not isinstance(tag, str):
        raise TypeError(f"Tag '{tag}' should be a string.")
        
    # drift_result 변수가 사전형(dict)인지 확인
    if not isinstance(drift_result, dict):
        raise TypeError("The 'drift_result' argument should be a dictionary.")
        
    # YAML 파일 로드
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # tag가 YAML 파일에 있는지 확인
    if tag not in config:
        raise ValueError(f"Tag '{tag}' is not found in the configuration file.")
        
    # drift_params 딕셔너리 초기화
    config[tag].setdefault('drift_params', {
        "average": 0,
        "cusum_threshold": 5,
        "ewma_alpha": 0.1,
        "ewma_smoothed": 0,
        "cusum_limit": 10,
        "cusum_plus": 0,
        "cusum_minus": 0
    })
    
    # drift_result 값을 drift_params 딕셔너리에 업데이트
    config[tag]['drift_params']['cusum_plus'] = drift_result['cusum_plus']
    config[tag]['drift_params']['cusum_minus'] = drift_result['cusum_minus']
    config[tag]['drift_params']['ewma_smoothed'] = drift_result['ewma_smoothed']

    # YAML 파일 저장
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
