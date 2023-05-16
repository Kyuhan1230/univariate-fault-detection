import json, traceback
from classify_fault.check_boundary import detect_out_of_bounds
from classify_fault.check_drift import detect_drift
from classify_fault.check_frozen import detect_frozen
from classify_fault.check_dynamics import detect_dynamics
from classify_fault.utils.get_value_in_dict import get_value
from classify_fault.set_config import update_config, set_boundary

def detect_fault(data, tracking_size, type_to_check=None, 
                 frozen_threshold=None, boundary_limits=None, 
                 dynamic_threshold=None, drift_params=None,
                 tag=None, config_path='./config/variable_config.json', 
                 boundary_type='fix'):
    """
    변수 유형을 확인하고, 각 유형의 장애를 감지하는 함수입니다.
    
    Args:
        data (list): 시계열 데이터의 2D 배열
        tracking_size (int): 최근 데이터의 크기
        type_to_check (dict): 각 유형의 장애를 확인할지 여부를 나타내는 딕셔너리
        frozen_threshold (float, optional): Frozen 검사에서 사용할 임계값. Defaults to None.
        boundary_limits (tuple, optional): Boundary 검사에서 사용할 상한값과 하한값. Defaults to None.
        dynamic_threshold (float, optional): Dynamic 검사에서 사용할 임계값. Defaults to None.
        drift_params (dict, optional): Drift 검사에서 사용할 평균, CUSUM 임계값, EWMA alpha값을 갖는 딕셔너리. Defaults to None.
    
    Returns:
        dict: 함수 실행 결과를 담은 딕셔너리. 다음과 같은 key를 포함합니다.
            - success (bool): 함수 실행 성공 여부
            - fault_detected (bool): 장애가 감지되었는지 여부
            - Frozen (bool): Frozen 장애가 감지되었는지 여부
            - Boundary (bool): Boundary 장애가 감지되었는지 여부
            - Dynamics (bool): Dynamics 장애가 감지되었는지 여부
            - Drift (bool): Drift 장애가 감지되었는지 여부
            - message (str): 예외 발생 시 예외 메시지
    
    Examples:
            import numpy as np
            # 예시 데이터
            data = np.array([1, 2, 2, 2, 10, 10])

            type_to_check = {
                "frozen": True,
                "boundary": True,
                "dynamics": True,
                "drift": True
            }

            frozen_threshold = 0.5
            tracking_size = 3
            boundary_limits = {"high": 12, "low":0 }
            dynamic_threshold = 2.5
            drift_params = {"average": 2, ", cusum_threshold": 4.0, "ewma_alpha": 0.2}

            fault_detected = detect_fault(data, tracking_size, type_to_check, frozen_threshold, boundary_limits, dynamic_threshold, drift_params)
            print(f"Is fault detected? {fault_detected['fault_detected']}")
            >>> Is fault detected? False
    """
    fault_detected = False
    values = {"frozen": None, "boundary": None, "dynamics": None, "drift": None}
    frozen_detected, boundary_detected, dynamic_detected, drift_detected = False, False, False, False

    if type_to_check is None:
        type_to_check = {"frozen": True, "boundary": True, "dynamics": True, "drift": True}

    try:
        # Frozen Test
        if type_to_check.get("frozen"):
            frozen_detected, avg_diff = detect_frozen(data, frozen_threshold, tracking_size)
            values['frozen'] = avg_diff
            if frozen_detected:
                fault_detected = True

        # Boundary Test
        statistics_update = None
        boundary_limits_update = None
        if type_to_check.get("boundary"):
            x = data[-1]  # 가장 최근 데이터
            high, low = boundary_limits['high'], boundary_limits['low']

            result = detect_out_of_bounds(x, high, low)
            boundary_detected = result["result"][0]
            values['boundary'] = result['result']
            if boundary_detected:
                fault_detected = True

            if 'moving' in boundary_type.lower():
                statistics_update, boundary_limits_update = {}, {}
                # JSON 파일 로드
                with open(config_path, "r") as f:
                    config = json.load(f)
                statistics = config[tag]['statistic']
                high_updated, low_updated, avg_updated, std_updated = set_boundary(statistics=statistics, x=data[-1], boundary_type='moving')['result']
                boundary_limits_update['high'] = high_updated
                boundary_limits_update['low'] = low_updated
                statistics_update['mean'] = avg_updated
                statistics_update['std'] = std_updated
                statistics_update['max'], statistics_update['min'] = max(max(data), statistics['max']), min(min(data), statistics['min'])
                statistics_update['oldest_value'] = data[0]

        # Dynamic Test
        if type_to_check.get("dynamics"):
            dynamic_detected, avg_diff = detect_dynamics(data=data, dynamic_threshold=dynamic_threshold)
            values['dynamics'] = avg_diff
            if dynamic_detected:
                fault_detected = True

        # Drift Test
        if type_to_check.get("drift"):
            data_point = data[-1]  # 가장 최근 데이터
            average, cusum_threshold, ewma_alpha = drift_params['average'], drift_params['cusum_threshold'], drift_params['ewma_alpha']
           
            cusum_plus_init = get_value(dictionary=drift_params, key='cusum_plus', default_value=0)
            cusum_minus_init = get_value(dictionary=drift_params, key='cusum_minus', default_value=0)
            result = detect_drift(data_point=data_point, average=average, cusum_threshold=cusum_threshold, ewma_alpha=ewma_alpha,
                                  C_plus=cusum_plus_init, C_minus=cusum_minus_init)

            cusum_plus=get_value(dictionary=result['CUSUM'], key='C_plus', default_value=0)
            cusum_minus=get_value(dictionary=result['CUSUM'], key='C_minus', default_value=0)
            ewma_smoothed=get_value(dictionary=result['EWMA'], key='smoothed_data', default_value=0)

            drift_params_update = {"cusum_plus": cusum_plus, 
                                   "cusum_minus": cusum_minus,
                                   "ewma_smoothed": ewma_smoothed}
            # Update Values Dictionary
            values['drift'] = [cusum_plus, cusum_minus]
            # if result["CUSUM"]["detected"] or result["EWMA"]["detected"]:
            if result["CUSUM"]["detected"]:
                drift_detected = True
                fault_detected = True
        else:
            drift_params_update = None

        # Update Configuration
        update_config(config_path=config_path, tag=tag, 
                      drift_params=drift_params_update, statistic=statistics_update, boundary_limits=boundary_limits_update)

        success = True
        message = "-"
    except Exception as E:
        success = False
        message = f"{E},\n {traceback.format_exc()}"
    
    result = {"success": success, "fault_detected": fault_detected, 
              "Frozen": frozen_detected, "Boundary": boundary_detected, "Dynamics": dynamic_detected, "Drift": drift_detected,
              "message": message, "values": values}
    
    return result  
