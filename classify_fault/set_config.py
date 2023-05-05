import json
import numpy as np
import pandas as pd


def calculate_variables_config(tag_list: list, data, type_to_check=None, 
                               frozen_threshold=0.01, tracking_size=5, 
                               boundary_limits=None, drift_params=None):
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

    staticstic = get_statistics(data=data, columns_list=tag_list)
    if staticstic['success'] is False:
        raise ValueError("Can't Calcualte Statistic. Something is wrong.")

    variables_config = {}
    for i, tag in enumerate(tag_list):
        mean = float(np.mean(data[:, i]))
        std = float(np.std(data[:, i]))

        if boundary_limits[i] is None:
            sigma_multiplier = 3
            while sigma_multiplier <= 6:
                high_limit = mean + sigma_multiplier * std
                low_limit = mean - sigma_multiplier * std
                
                within_limits = np.sum((data[:, i] >= low_limit) & (data[:, i] <= high_limit))
                data_count = len(data[:, i])
                confidence_percentage = within_limits / data_count
                
                if confidence_percentage >= 0.99:
                    break
                
                sigma_multiplier += 0.5
            
            boundary_limits[i] = {"high": high_limit, "low": low_limit}

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

        variables_config[tag] = {
            "tag_name": tag,
            "type_to_check": type_to_check,
            "frozen_threshold": frozen_threshold,
            "tracking_size": tracking_size,
            "boundary_limits": boundary_limits[i],
            "dynamic_threshold": 0.5,
            "drift_params": drift_params[i],
            "statistic": staticstic["result"][tag]
        }

    return variables_config


def get_statistics(data, columns_list=None):
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
        data_size = int(column_data.size)

        statistics[column_name] = {"mean": mean, "std": std, "median": median, "quantile1": q1, "quantile3": q3,
                                    "iqr": iqr, "min": min_val, "max": max_val, "oldest_value": oldest_value,
                                    "data_size": data_size}
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
