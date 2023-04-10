import yaml
import numpy as np


def calculate_variables_config(tag_list: list, data, type_to_check=None, frozen_threshold=0.01, tracking_size=5, boundary_limits=None, drift_params=None):
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

    variables_config = {}
    for i, tag in enumerate(tag_list):
        mean = float(np.mean(data[:, i]))
        std = float(np.std(data[:, i]))

        if boundary_limits[i] is None:
            boundary_limits[i] = (mean + 3 * std, mean - 3 * std)

        if drift_params[i] is None:
            drift_params[i] = {
                "average": mean,
                "cusum_threshold": 5 * std * 3 / 2,
                "ewma_alpha": 0.1
            }

        variables_config[tag] = {
            "tag_name": tag,
            "type_to_check": type_to_check,
            "frozen_threshold": frozen_threshold,
            "tracking_size": tracking_size,
            "boundary_limits": boundary_limits[i],
            "dynamic_threshold": 0.5,
            "drift_params": drift_params[i]
        }

    return variables_config


def save_config(data, yaml_file_path='./config/variable_config.yaml'):
    """
    주어진 데이터를 YAML 형식으로 저장합니다.
    
    Args:
        data (dict): 변수 설정값을 담고 있는 딕셔너리
        yaml_file_path (str, optional): 저장할 YAML 파일 경로, default는 './config/variable_config.yaml'
    """
    # YAML 형식으로 변환합니다.
    yaml_data = yaml.dump(data, default_flow_style=False)

    # 파일로 저장합니다.
    with open(yaml_file_path, 'w') as f:
        f.write(yaml_data)


def load_config(yaml_file_path='./config/variable_config.yaml'):
    """
    주어진 YAML 파일을 로드합니다.
    
    Args:
        yaml_file_path (str, optional): 로드할 YAML 파일 경로, default는 './config/variable_config.yaml'
    
    Returns:
        dict: 변수 설정값을 담고 있는 딕셔너리
    """
    # YAML 파일을 읽어와 딕셔너리로 변환합니다.
    with open(yaml_file_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data


tag_list = ["Var1", "Var2"]
data = np.array([
            [1, 10],
            [2, 9],
            [3, 11],
            [4, 10],
            [5, 12],
            [6, 9]
        ])
