from classify_fault.__init__ import np, pd, traceback, warnings, Dict, Union, List, Tuple
from classify_fault.check_boundary import detect_out_of_bounds, detect_out_of_bounds_vec
from classify_fault.check_drift import detect_drift, detect_drift_vec
from classify_fault.check_frozen import detect_frozen, detect_frozen_vec
from classify_fault.check_dynamics import detect_dynamics, detect_dynamics_vec
from classify_fault.utils.get_value_in_dict import get_value
from classify_fault.utils.validate_type import all_elements_equal, validate_data
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

def detect_faults_v2(data: Union[np.ndarray, pd.DataFrame],
                  tag_list: Union[list, tuple],
                  config_path: str = './config/variable_config.json') -> dict:
    try:
        data, tag_list = validate_data(data, tag_list)
    except (TypeError, ValueError) as e:
        return {"success": False, "message": str(e)}

    var_num, data_size = len(tag_list), len(data)

    results, updates = {}, {}
    config = load_config(json_file_path=config_path)

    tracking_size_list = [config[tag]['tracking_size'] for tag in tag_list]
    if max(tracking_size_list) > data_size:
        raise ValueError("The tracking size cannot be larger than the maximum size of the input data.")

    frozen_threshold_list = [config[tag]['frozen_threshold'] for tag in tag_list]
    dynamic_threshold_list = [config[tag]['dynamic_threshold'] for tag in tag_list]
    high_list = [config[tag]['boundary_limits']['high'] for tag in tag_list]
    low_list = [config[tag]['boundary_limits']['low'] for tag in tag_list]
    boundary_type_list = [config[tag]['statistic']['boundary_type'] for tag in tag_list]
    drift_params_list = [config[tag]['drift_params'] for tag in tag_list]

    try:
        if all_elements_equal(tracking_size_list):
            dynamic_threshold_list = np.array(dynamic_threshold_list)
            frozen_threshold_list = np.array(frozen_threshold_list)

            dynamic_detected_results, avg_diffs = detect_dynamics_vec(data, dynamic_threshold_list)
            frozen_detected_results, avg_diffs = detect_frozen_vec(data, frozen_threshold_list)
        else:
            dynamic_detected_results, frozen_detected_results, avg_diffs = [], [], []
            for i, tag in enumerate(tag_list):
                dynamic_detected_results[i], avg_diffs[i] = detect_dynamics(data=data[:, i],
                                                                            dynamic_threshold=dynamic_threshold_list[i])
                frozen_detected_results[i], avg_diffs[i] = detect_frozen(data=data[:, i],
                                                                         frozen_threshold=dynamic_threshold_list[i])
            dynamic_detected_results = np.array(dynamic_detected_results)
            frozen_detected_results = np.array(frozen_detected_results)
            avg_diffs = np.array(avg_diffs)
    except Exception as E:
        return {"success": False, "message": str(f"{E},\n {traceback.format_exc()}")}

    try:
        target = data[-1, :].reshape(1, -1)
        high_array = np.array(high_list).reshape(1, var_num)
        low_array = np.array(low_list).reshape(1, var_num)
        bound_result = detect_out_of_bounds_vec(x=target, high=high_array, low=low_array)

        statistics_updates, boundary_limits_updates = [], []
        for i, (boundary_type, tag) in enumerate(zip(boundary_type_list, tag_list)):
            statistics_update, boundary_limits_update = None, None
            if 'moving' in boundary_type.lower():
                statistics = config[tag]['statistic']
                boundary_output = set_boundary(statistics=statistics,
                                               x=data[-1, i],
                                               boundary_type='moving')['result']
                boundary_limits_update = {'high': boundary_output[0], 'low': boundary_output[1]}
                statistics_update = {
                    'mean': boundary_output[2],
                    'std': boundary_output[3],
                    'max': max(max(data[:, i]), statistics['max']),
                    'min': min(min(data[:, i]), statistics['min']),
                    'oldest_value': data[0, i],
                    'boundary_type': 'moving',
                    'tracked_size': boundary_output[4]
                }
            statistics_updates.append(statistics_update)
            boundary_limits_updates.append(boundary_limits_update)
    except Exception as e:
        return {"success": False, "message": str(f"{e},\n {traceback.format_exc()}")}

    try:
        df = pd.DataFrame(drift_params_list)
        drift_result = detect_drift_vec(data_points=target, averages=df["average"].values,
                                        cusum_thresholds=df["cusum_threshold"].values,
                                        ewma_alphas=df["ewma_alpha"].values,
                                        C_pluses=df["cusum_plus"].values, C_minuses=df["cusum_minus"].values,
                                        smoothed_data=df["ewma_smoothed"].values, cusum_limit=df["cusum_limit"].values)
        C_pluses = drift_result['CUSUM']['C_plus'][0]
        C_minuses = drift_result['CUSUM']['C_minus'][0]
        smoothed_data = drift_result['EWMA']['smoothed_data'][0]
    except Exception as e:
        return {"success": False, "message": str(f"{e},\n {traceback.format_exc()}")}
    
    for i, tag in enumerate(tag_list):
        boundary_detected = bound_result['result'][0][0][i]
        drift_detected = drift_result["CUSUM"]["detected"][0][i]
        dynamic_detected = dynamic_detected_results[i]
        frozen_detected = frozen_detected_results[i]
        # bound_result: {'success': np.array([True, True]), 'result': [np.array([False, False]), np.array([10.5, 8.5])]}
        fault_detection = {"success": True,
                           "values": {'boundary': [boundary_detected, bound_result['result'][1][0][i]], 
                                      'drift': [C_pluses[i], C_minuses[i]], 
                                      'dynamic': avg_diffs[i], 
                                      'frozen': avg_diffs[i]},
                           "fault_detected": any([boundary_detected, drift_detected, dynamic_detected, frozen_detected]),
                           "Frozen": frozen_detected, "Boundary": boundary_detected, "Dynamics": dynamic_detected,
                           "Drift": drift_detected,
                           "statistics_update": statistics_updates[i],
                           "boundary_limits_update": boundary_limits_updates[i],
                           "drift_params_update": {"cusum_plus": C_pluses[i],
                                                   "cusum_minus": C_minuses[i],
                                                   "ewma_smoothed": smoothed_data[i]},
                           "message": '-',
                           }

        results[tag] = fault_detection
        updates[tag] = {"drift_params": fault_detection["drift_params_update"],
                        "statistic": fault_detection["statistics_update"],
                        "boundary_limits": fault_detection["boundary_limits_update"]}

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
            dynamic_detected, avg_diff = detect_dynamics(data=data, dynamic_threshold=dynamic_threshold, n=tracking_size)
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
        success = False
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
