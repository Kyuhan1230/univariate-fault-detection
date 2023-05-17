from classify_fault.__init__ import *
from classify_fault.check_boundary import detect_out_of_bounds
from classify_fault.check_drift import detect_drift
from classify_fault.check_frozen import detect_frozen
from classify_fault.check_dynamics import detect_dynamics
from classify_fault.utils.get_value_in_dict import get_value
from classify_fault.set_config import update_config, set_boundary, load_config

def detect_fault(data, tracking_size, type_to_check=None, 
                 frozen_threshold=None, boundary_limits=None, 
                 dynamic_threshold=None, drift_params=None,
                 tag=None, config_path:str=None,
                 boundary_type='fix'):
    # Set Parameter
    results, updates = {}, {}

    # load config
    if config_path is not None:
        config = load_config(json_file_path=config_path)
    else:
        config = None

    # Perform fault detection
    fault_detection = perform_fault_detection(data, tracking_size, config=config,
                                              type_to_check=type_to_check, 
                                              frozen_threshold=frozen_threshold, boundary_limits=boundary_limits, 
                                              dynamic_threshold=dynamic_threshold, drift_params=drift_params,
                                              tag=tag, boundary_type=boundary_type)
    # Gather results and updates
    # results[tag] = fault_detection
    updates[tag] = {"drift_params": fault_detection["drift_params_update"], 
                    "statistic": fault_detection["statistics_update"], 
                    "boundary_limits": fault_detection["boundary_limits_update"]}

    # Update Configuration
    if config_path is not None and config is not None:
        update_config(config_path=config_path, updates=updates)
    
    return fault_detection


def detect_faults(data, tag_list:list, config_path='./config/variable_config.json'):
    # data 유효성 검사
    if not isinstance(data, (np.ndarray, pd.DataFrame)):
        raise TypeError("Data Should be Numpy ndarray or DataFrame")
    else:
        if isinstance(data, pd.DataFrame) and tag_list is None:
            tag_list = data.columns.to_list()
            data = data.values

    # data shape 검사
    if data.ndim == 1:
        warnings.warn("For one tag, use the 'detect_fault' function.", UserWarning)

    # tag_list 유효성 검사
    if not isinstance(tag_list, (list, tuple)):
        raise TypeError("tag list should be list. ex) tag_list=['tag01', 'tag2', ..., 'tag10']")
    if isinstance(tag_list, tuple):
        warnings.warn("Tag ListPlease use the tag list as a list, not as a tuple.", UserWarning)
        tag_list = list(tag_list)       
    
    # Set Parameter
    results, updates = {}, {}

    # config_path 읽기
    config = load_config(json_file_path=config_path)

    for i, tag in enumerate(tag_list):
        # 필요 파라미터 읽기
        tracking_size = config[tag]['tracking_size']
        type_to_check = config[tag]['type_to_check']
        frozen_threshold = config[tag]['frozen_threshold']
        boundary_limits = config[tag]['boundary_limits']
        dynamic_threshold = config[tag]['dynamic_threshold']
        drift_params = config[tag]['drift_params']
        boundary_type = config[tag]['statistic']['boundary_type']

        target = data[:, i]

        fault_detection = perform_fault_detection(data=target, tracking_size=tracking_size, config=config,
                                                  type_to_check=type_to_check,
                                                  frozen_threshold=frozen_threshold, boundary_limits=boundary_limits, 
                                                  dynamic_threshold=dynamic_threshold, drift_params=drift_params,
                                                  tag=tag, boundary_type=boundary_type)
        # Gather results and updates
        results[tag] = fault_detection
        updates[tag] = {"drift_params": fault_detection["drift_params_update"], 
                        "statistic": fault_detection["statistics_update"], 
                        "boundary_limits": fault_detection["boundary_limits_update"]}
    
    # Update Configuration for all tags at once
    update_config(config_path=config_path, updates=updates)

    return results    


def perform_fault_detection(data, tracking_size, config=None, 
                            type_to_check=None, 
                            frozen_threshold=None, boundary_limits=None, 
                            dynamic_threshold=None, drift_params=None,
                            tag=None, boundary_type='fix'):
    fault_detected = False
    values = {"frozen": None, "boundary": None, "dynamics": None, "drift": None}
    frozen_detected, boundary_detected, dynamic_detected, drift_detected = False, False, False, False

    if type_to_check is None:
        type_to_check = {"frozen": True, "boundary": True, "dynamics": True, "drift": True}

    if tracking_size == 0:
        type_to_check = {"frozen": False, "boundary": True, "dynamics": False, "drift": True}

    try:
        # Frozen Test
        if type_to_check.get("frozen"):
            frozen_detected, avg_diff = detect_frozen(data, frozen_threshold, tracking_size)
            values['frozen'] = avg_diff
            if frozen_detected:
                fault_detected = True

        # Boundary Test
        boundary_limits_update, statistics_update = None, None

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
                statistics = config[tag]['statistic']
                high_updated, low_updated, avg_updated, std_updated, tracked_size = set_boundary(statistics=statistics, 
                                                                                                 x=data[-1], 
                                                                                                 boundary_type='moving')['result']
                boundary_limits_update['high'] = high_updated
                boundary_limits_update['low'] = low_updated
                statistics_update['mean'] = avg_updated
                statistics_update['std'] = std_updated
                statistics_update['max'], statistics_update['min'] = max(max(data), statistics['max']), min(min(data), statistics['min'])
                statistics_update['oldest_value'] = data[0]
                statistics_update['boundary_type'] = 'moving'
                statistics_update['tracked_size'] = tracked_size

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
        
        success = True
        message = "-"

    except Exception as E:
        success = True
        message = f"{E},\n {traceback.format_exc()}"

    fault_detection = {"success": success,
                       "values": values,
                       "fault_detected": fault_detected,
                       "Frozen": frozen_detected,
                       "Boundary": boundary_detected,
                       "Dynamics": dynamic_detected,
                       "Drift": drift_detected,
                       "statistics_update": statistics_update,
                       "boundary_limits_update": boundary_limits_update,
                       "drift_params_update": drift_params_update,
                       "message": message}
    
    return fault_detection