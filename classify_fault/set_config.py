import os
import json
import numpy as np
import pandas as pd
from typing import List


def calculate_variables_config(tag_list: list, data, type_to_check=None, 
                               frozen_threshold=0.01, tracking_size=5, 
                               boundary_limits=None, drift_params=None,
                               boundary_type='Fix'):
    """
    주어진 데이터를 바탕으로 변수 설정값을 계산합니다.
    
    Args:
        tag_list (list): 변수 이름을 담고 있는 리스트
        data (ndarray): 시계열 데이터
        type_to_check (dict, optional): 검사 유형을 지정하는 딕셔너리, default는 None
        frozen_threshold (float, optional): Frozen 장애를 검출하기 위한 임계값, default는 0.01
        n (int, optional): Boundary 장애 검출 시 사용될 데이터 수, default는 5
        boundary_limits (list, optional): 상한값 및 하한값을 지정하는 튜플의 리스트, default는 None
        drift_params (list, optional): Drift 장애 검출 시 사용될 데이터를 지정하는 딕셔너리의 리스트, default는 None
    
    Returns:
        dict: 변수 이름을 키로, 설정값을 값으로 갖는 딕셔너리
    
    Examples:
        tag_list = ["Var1", "Var2"]
        data = np.array([
            [1, 10],
            [2, 9],
            [3, 11],
            [4, 10],
            [5, 12],
            [6, 9]
        ])
        variables_config = calculate_variables_config(tag_list, data)
    """
    if type_to_check is None:
        type_to_check = {"frozen": True, "boundary": True, "dynamics": True, "drift": True}
    if boundary_limits is None:
        boundary_limits = [None] * len(tag_list)
    if drift_params is None:
        drift_params = [None] * len(tag_list)

    staticstic = get_statistics(data=data, columns_list=tag_list, tracking_size=tracking_size)
    if staticstic['success'] is False:
        raise ValueError("Can't Calcualte Statistic. Something is wrong.")

    variables_config = {}
    for i, tag in enumerate(tag_list):
        mean = float(staticstic["result"][tag]['mean'])
        std = float(staticstic["result"][tag]['std'])

        if boundary_limits[i] is None:
            boundary_res = set_boundary(statistics=staticstic['result'][tag], x=float(data[-1, i]), 
                                        boundary_type=boundary_type, data=data[:, i].ravel())
            boundary_limits[i] = {"high": boundary_res['result'][0], "low": boundary_res['result'][1]}

        if drift_params[i] is None:
            drift_params[i] = {
                "average": mean,
                "cusum_threshold": 5 * std * 1 / 2,
                "ewma_alpha": 0.1,
                "ewma_smoothed": 0,
                "cusum_limit": 5,
                "cusum_plus": 0,
                "cusum_minus": 0
            }
        
        # dynamic_threshold를 결정
        dynamic_threshold = determine_dynamic_threshold(data[:, i])

        variables_config[tag] = {
            "tag_name": tag,
            "type_to_check": type_to_check,
            "frozen_threshold": frozen_threshold,
            "tracking_size": tracking_size,
            "boundary_limits": boundary_limits[i],
            "dynamic_threshold": dynamic_threshold,
            "drift_params": drift_params[i],
            "statistic": staticstic["result"][tag]
        }

    return variables_config


def get_statistics(data, columns_list=None, tracking_size=5):
    """
    주어진 데이터를 바탕으로 데이터의 통계치 dictionary를 완성합니다.
    
    Args:
        data (list or np.ndarray or pd.DataFrame): 분석할 데이터
        columns_list (list, optional): 분석할 칼럼 이름 리스트
    
    Returns:
        dict: 데이터의 통계치를 담은 딕셔너리
            - success (bool): 함수 실행 결과를 나타내는 값
                처리가 제대로 이루어졌으면 True, 그렇지 않으면 False
            - result (dict): 데이터의 통계치를 담은 딕셔너리
                - mean: 평균값
                - std: 표준편차
                - median: 중앙값
                - quantile1: 1사분위수
                - quantile3: 3사분위수
                - iqr: 사분위 범위(iqr)
                - min: 최소값
                - max: 최대값
                - oldest_value: 가장 오래된 데이터 값
                - data_size: 데이터 크기
    """
    if not isinstance(data, (list, np.ndarray, pd.DataFrame)):
        raise ValueError(f"Invalid data type. Expected list or np.ndarray or pd.DataFrame, but got {type(data)}")

    if isinstance(data, pd.DataFrame):
        if columns_list is None:
            columns_list = data.columns.tolist()
        data = data.values

    statistics = {}

    for i, column_name in enumerate(columns_list):
        column_data = data[:, i]
        mean = float(np.mean(column_data))
        std = float(np.std(column_data, ddof=1))
        median = float(np.median(column_data))
        q1 = float(np.quantile(column_data, 0.25))
        q3 = float(np.quantile(column_data, 0.75))
        iqr = q3 - q1
        min_val = float(np.min(column_data))
        max_val = float(np.max(column_data))
        oldest_value = float(column_data[0])

        statistics[column_name] = {"mean": mean, "std": std, "median": median, "quantile1": q1, "quantile3": q3,
                                    "iqr": iqr, "min": min_val, "max": max_val, "oldest_value": oldest_value,
                                    "tracking_size": tracking_size, 'boundary_type': 'fix'}
    return {"success": True, "result": statistics}


def save_config(data, json_file_path='./config/variable_config.json'):
    """
    주어진 데이터를 JSON 형식으로 저장합니다.
    
    Args:
        data (dict): 변수 설정값을 담고 있는 딕셔너리
        json_file_path (str, optional): 저장할 JSON 파일 경로, default는 './config/variable_config.json'
    """
    # JSON 형식으로 변환합니다.
    json_data = json.dumps(data, indent=4)

    # 파일로 저장합니다.
    with open(json_file_path, 'w') as f:
        f.write(json_data)


def load_config(json_file_path='./config/variable_config.json'):
    """
    주어진 JSON 파일을 로드합니다.
    
    Args:
        json_file_path (str, optional): 로드할 JSON 파일 경로, default는 './config/variable_config.json'
    
    Returns:
        dict: 변수 설정값을 담고 있는 딕셔너리
    """
    # JSON 파일을 읽어와 딕셔너리로 변환합니다.
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    return data


def determine_dynamic_threshold(data: List[float], sensitivity: float = 1.5) -> float:
    """
    주어진 데이터를 바탕으로 동적 임계값을 결정합니다.

    Args:
        data (list): 안정상태를 검사할 이전 데이터의 리스트.
        sensitivity (float): 동적 임계값 조절을 위한 감도 값.

    Returns:
        float: 동적 임계값.

    Examples:
        data = [1, 2, 4, 6, 8, 10, 12]
        window_size = 5
        sensitivity = 1.5

        dynamic_threshold = determine_dynamic_threshold(data, window_size, sensitivity)
        print(f"Dynamic threshold: {dynamic_threshold}")
        # Dynamic threshold: 3.0
    """
    data = np.array(data)
    data_diff = np.diff(data)  # Calculate differences between consecutive elements

    moving_average = np.mean(data_diff)
    moving_std = np.std(data_diff)

    dynamic_threshold = moving_average + (moving_std * sensitivity)

    return float(dynamic_threshold)

def set_boundary(statistics, x=None, boundary_type='fix', data=None):
    """
    주어진 통계 정보와 경계 유형, 입력값, 시그마 레벨, 태그 이름을 사용하여 경계 값을 계산합니다.
    
    Args:
        statistics (dict): 통계 정보를 저장하는 딕셔너리
        x (float): 현재 데이터
        sigma_level (float): 시그마 레벨
        boundary_type (str): 경계 유형을 뜻하는 문자열, 기본 값: 'fix'
        data (ndarray): 시계열 데이터, 기본 값: None
    
    Returns:
        dict: 경계 값을 담은 딕셔너리
            - success (bool): 함수 실행 결과를 나타내는 값입니다. 경계 값을 계산할 수 있으면 True를 반환
            - result (list): 경계 값을 담은 리스트입니다. 계산에 실패하면 None을 반환
    """        
    # boundary_type 인자가 문자열 타입인지 확인
    if not isinstance(boundary_type, str):
        raise TypeError("The 'boundary_type' argument must be a string.")
    
    valid_boundary_types = ["moving", "fix"]
    boundary_type = boundary_type.lower().replace(" ", "_")

    if boundary_type not in valid_boundary_types:
        raise ValueError(f"Invalid boundary type: {boundary_type}. Valid values should has {valid_boundary_types}.")
    
    # Statistics 인자가 딕셔너리 타입인지 확인
    if not isinstance(statistics, dict):
        raise TypeError("The 'statistics' argument must be a dictionary.")
    
    # Statistics 인자 내 mean, std 값 추출
    mean, std = statistics['mean'], statistics['std']

    # 4. 경계 값 계산
    if 'fix' in boundary_type:
        if data is None:
            raise ValueError("For 'fix' bounds, the data argument must be entered.")
        if not isinstance(data, np.ndarray) or (data.ndim > 1):
            raise TypeError("The 'data' arguement must be 1d ndarray.")
        
        sigma_levels = np.arange(3, 6.5, 0.5)
        high_limits = mean + sigma_levels * std
        low_limits = mean - sigma_levels * std
        
        within_limits = np.sum((data[:, None] >= low_limits) & (data[:, None] <= high_limits), axis=0)
        data_count = len(data)
        confidence_percentages = within_limits / data_count

        best_index = np.argmax(confidence_percentages >= 0.99)
        best_sigma_level = sigma_levels[best_index]
        
        high = mean + best_sigma_level * std
        low = mean - best_sigma_level * std
        return {"result": [high, low, mean, std]}

    else:    # "moving" in boundary_type
        # x 인자가 실수 타입인지 확인
        if x is None or not isinstance(x, float):
            raise TypeError("The 'x' argument representing the current value must be floating point.")

        try:
            from classify_fault.update_avg_std import update_avg_std
        except ImportError:
            raise ImportError("The 'update_avg_std' function is required but could not be imported.")
        avg_old = mean
        std_old = std
        update_option = 'Keep Size'
        oldest_value = statistics['oldest_value']
        data_size = statistics['tracking_size']

        avg_updated, std_updated = update_avg_std(avg_old=avg_old, std_old=std_old, new_value=x, 
                                                  update_option=update_option, 
                                                  oldest_value=oldest_value, data_size=data_size)

        statistics['mean'] = avg_updated
        statistics['std'] = std_updated
        
        high = avg_updated + 1.5 * std_updated
        low = avg_updated - 1.5 * std_updated
        return {"result": [high, low, avg_updated, std_updated]}
    

def update_config(config_path: str, tag: str, config=None, drift_params: dict=None, statistic: dict=None, boundary_limits: dict=None):
    """주어진 config 파일 내의 특정 tag의 drift_params와 statistic을 업데이트하고 저장합니다.

    Args:
        config_path (str): JSON 파일 경로
        tag (str): 업데이트할 대상 tag 이름
        drift_params (dict): 업데이트할 drift_params 값들이 포함된 딕셔너리
        statistic (dict): 업데이트할 statistic 값들이 포함된 딕셔너리

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        TypeError: tag가 문자열이 아닌 경우 또는 config 파일이 JSON 형식이 아닌 경우
        ValueError: tag가 config 파일에 존재하지 않는 경우

    """
    if tag is None:
        print('tag is None')
        return 

    # tag가 문자열인지 확인
    if not isinstance(tag, str):
        raise TypeError(f"Tag '{tag}' should be a string.")
        
    # 파일 타입이 json인지 확인
    if not config_path.endswith('.json'):
        raise TypeError("Currently only JSON configuration can be updated and other data types require separate update.")

    # Config JSON 파일 로드
    if config is None:
        with open(config_path, "r") as f:
            config = json.load(f)
        
    # tag가 JSON 파일에 있는지 확인
    if tag not in config:
        raise ValueError(f"Tag '{tag}' is not found in the configuration file.")
        
    # drift_result 값을 drift_params 딕셔너리에 업데이트
    if drift_params is not None:
        # drift_params와 statistic 딕셔너리 초기화
        config[tag].setdefault('drift_params', {
            "average": 0,
            "cusum_threshold": 5,
            "ewma_alpha": 0.1,
            "ewma_smoothed": 0,
            "cusum_limit": 10,
            "cusum_plus": 0,
            "cusum_minus": 0
        })
        config[tag]['drift_params']['cusum_plus'] = drift_params['cusum_plus']
        config[tag]['drift_params']['cusum_minus'] = drift_params['cusum_minus']
        config[tag]['drift_params']['ewma_smoothed'] = drift_params['ewma_smoothed']
    
    # statistic 값을 statistic 딕셔너리에 업데이트
    if statistic is not None:
        config[tag].setdefault('statistic', {
        "mean": 0,
        "std": 0,
        "median": 0,
        "quantile1": 0,
        "quantile3": 0,
        "iqr": 0,
        "min": 0,
        "max": 0,
        "oldest_value": 0,
        "tracking_size": 0
        })
        config[tag]['statistic']['mean'] = statistic['mean']
        config[tag]['statistic']['std'] = statistic['std']
        config[tag]['statistic']['min'] = statistic['min']
        config[tag]['statistic']['max'] = statistic['max']
        config[tag]['statistic']['oldest_value'] = statistic['oldest_value']
    
    # boundary_limits 값을 boundary_limits 딕셔너리에 업데이트
    if boundary_limits is not None:
        config[tag].setdefault('boundary_limits', {
        "high": 0,
        "low": 0,
        })
        config[tag]['boundary_limits']['high'] = boundary_limits['high']
        config[tag]['boundary_limits']['low'] = boundary_limits['low']
    # JSON 파일 저장
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)