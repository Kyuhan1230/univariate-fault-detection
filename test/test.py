import os
import sys, time
import numpy as np
import pandas as pd
from copy import deepcopy

sys.path.append(os.path.abspath('./'))

from classify_fault.set_config import calculate_variables_config, save_config, load_config
from classify_fault.fault_detection import detect_fault, detect_faults, detect_faults_v2
from classify_fault.utils.match_value import get_matching_keys

# 훈련 데이터와 테스트 데이터를 불러옵니다.
data_path = './data/HS_F1_sensor_raw_100.csv'
data_path2 = './data/HS_F1_sensor_raw_100.csv'

train_data = pd.read_csv(data_path, index_col=[0])
train_data = train_data.iloc[:int(len(train_data)*0.4), :]    # 40% 데이터를 이용하여 통계치 생성
test_data = pd.read_csv(data_path2, index_col=[0])

tag_list = train_data.columns.to_list()
test_tag_list = test_data.columns.to_list()
tracking_size = 1440    # 임의 설정

def fault_detection_with_config_for_1_point():
    tag_list = ["Var1", "Var2"]
    data = np.array([
        [1, 10],
        [1.2, 9],
        [1.3, 11],
        [0.94, 10],
        [1.05, 12],
        [0.96, 9]
    ])
    variables_config = calculate_variables_config(tag_list, data)

    save_config(variables_config, "./config/test_config.json")
    loaded_config = load_config("./config/test_config.json")

    test_data = np.array([[5, 20]])
    
    data_to_test_ = np.concatenate((data[:, 0], test_data[:, 0]))
    result = detect_fault(data=data_to_test_, 
                          tracking_size=loaded_config["Var1"]["tracking_size"], 
                          type_to_check=loaded_config["Var1"]["type_to_check"],
                          frozen_threshold=loaded_config["Var1"]["frozen_threshold"], 
                          boundary_limits=loaded_config["Var1"]["boundary_limits"],
                          dynamic_threshold=loaded_config["Var1"]["dynamic_threshold"], 
                          drift_params=loaded_config["Var1"]["drift_params"])

    assert result["success"], f"Error occurred: {result['message']}"
    assert result["fault_detected"]
    assert result["Frozen"]==False
    assert result["Boundary"]
    assert result["Dynamics"]
    assert result["Drift"]==False
    # 파일 삭제
    os.remove("./config/test_config.json")


def fault_detection_for_1_point():
    data = np.array([1, 1.2, 1.3, 0.94, 1.05, 0.96, 5])
    type_to_check = {
        "frozen": True,
        "boundary": True,
        "dynamics": True,
        "drift": True
    }
    frozen_threshold = 0.5
    tracking_size = 3
    boundary_limits = {"high": 4, "low": 0}
    dynamic_threshold = 0.5
    drift_params = {"average": 2, "cusum_threshold": 4.0, "ewma_alpha": 0.2}

    result = detect_fault(data, tracking_size, type_to_check, frozen_threshold, boundary_limits, dynamic_threshold, drift_params)

    assert result["success"] == True, f"Error occurred: {result['message']}"
    assert result["fault_detected"] == True, f"Error occurred: {result}"
    assert result["Frozen"] == False, f"Error occurred: {result}"
    assert result["Boundary"] == True, f"Error occurred: {result}"
    assert result["Dynamics"] == True, f"Error occurred: {result}"
    assert result["Drift"] == False, f"Error occurred: {result}"


def fault_detection_multiple_variable_with_config_for_1_point():
    tag_list = ["Var1", "Var2"]
    data = np.array([
        [1, 10],
        [1.2, 9],
        [1.3, 11],
        [0.94, 10],
        [1.05, 12],
        [0.96, 9]
    ])

    variables_config = calculate_variables_config(tag_list, data)

    save_config(variables_config, "./config/test_config.json")

    test_data = np.array([[5, 18]])
    
    data_to_test_ = np.concatenate((data, test_data), axis=0)
    result = detect_faults(data=data_to_test_, tag_list=tag_list, config_path="./config/test_config.json")
    # print(result)
    for tag in tag_list:
        assert result[tag]["success"], f"Error occurred: {result[tag]['message']}"
        assert result[tag]["fault_detected"]
        assert result[tag]["Frozen"]==False
        assert result[tag]["Boundary"]
        assert result[tag]["Dynamics"]
        assert result[tag]["Drift"]==False
    # 파일 삭제
    os.remove("./config/test_config.json")

def fault_detection_multiple_variable_with_config_for_several_pointsV1():
    
    config_save_path = './config/example_configAll_multiFaults.json'

    # 여러 변수에 대한 테스트를 진행하기 위하여 Configuration을 중복해서 작성하였습니다.
    config = calculate_variables_config(tag_list=test_tag_list, data=train_data.values)

    # Moving Boundary를 적용합니다.
    for tag in test_tag_list:
        config[tag]['statistic']['boundary_type'] = 'moving'

    # Configuration을 저장합니다.
    save_config(data=config, json_file_path=config_save_path)

    # Set Tracking Size
    tracking_size = 1440
    from classify_fault.utils.match_value import get_matching_keys

    tags = {tag: [] for tag in test_tag_list}
    detecteds, frz_res, bounds_res, dyn_res, drft_res, high_boundary, low_boundary = [deepcopy(tags) for _ in range(7)]

    start = time.time()
    for i in range(tracking_size, 10000):  # len(test_data)
        test_ = test_data[test_tag_list].values[i - tracking_size + 1: i + 1, :]
        res = detect_faults(data=test_, tag_list=test_tag_list, config_path=config_save_path)
        
        config = load_config(json_file_path=config_save_path)
        for tag in test_tag_list:
            result = get_matching_keys(res[tag]) or []

            frz_res[tag].append('frozen' in result)
            bounds_res[tag].append('boundary' in result)
            dyn_res[tag].append('dynamics' in result)
            drft_res[tag].append('drift' in result)

            detecteds[tag].append(bool(result))

            high_boundary[tag].append(config[tag]['boundary_limits']['high'])
            low_boundary[tag].append(config[tag]['boundary_limits']['low'])
    end = time.time()

    print(f"실행시간은 {(end - start)}sec 입니다.")
    
    # 파일 삭제
    os.remove(config_save_path)

def fault_detection_multiple_variable_with_config_for_several_pointsV2():
    
    config_save_path = './config/example_configAll_multiFaultsV2.json'

    # 훈련 데이터를 사용하여 Configuration을 계산합니다.
    # 여러 변수에 대한 테스트를 진행하기 위하여 Configuration을 중복해서 작성하였습니다.
    config = calculate_variables_config(tag_list=test_tag_list, data=train_data.values)

    tracking_size = 1440    # 임의 설정

    # Moving Boundary를 적용합니다.
    for tag in test_tag_list:
        config[tag]['statistic']['tracking_size'] = tracking_size
        config[tag]['statistic']['boundary_type'] = 'moving'

        config[tag]['tracking_size'] = tracking_size

    # Configuration을 저장합니다.
    save_config(data=config, json_file_path=config_save_path)

    from classify_fault.utils.match_value import get_matching_keys

    tags = {tag: [] for tag in test_tag_list}
    detecteds, frz_res, bounds_res, dyn_res, drft_res, high_boundary, low_boundary = [deepcopy(tags) for _ in range(7)]

    start = time.time()
    for i in range(tracking_size, 10000):  # len(test_data)
        test_ = test_data[test_tag_list].values[i - tracking_size + 1: i + 1, :]
        res = detect_faults_v2(data=test_, tag_list=test_tag_list, config_path=config_save_path)
        
        config = load_config(json_file_path=config_save_path)
        for tag in test_tag_list:
            result = get_matching_keys(res[tag]) or []

            frz_res[tag].append('frozen' in result)
            bounds_res[tag].append('boundary' in result)
            dyn_res[tag].append('dynamics' in result)
            drft_res[tag].append('drift' in result)

            detecteds[tag].append(bool(result))

            high_boundary[tag].append(config[tag]['boundary_limits']['high'])
            low_boundary[tag].append(config[tag]['boundary_limits']['low'])
    end = time.time()

    print(f"실행시간은 {(end - start)}sec 입니다.")
    
    # 파일 삭제
    os.remove(config_save_path)


def save_load_config():
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

    save_config(variables_config, "./config/test_config.json")
    loaded_config = load_config("./config/test_config.json")

    assert variables_config == loaded_config
    # 파일 삭제
    os.remove("./config/test_config.json")

if __name__ == "__main__":
    import cProfile
    import pstats
    from io import StringIO

    # save_load_config()
    # print("\033[43mTest: Save Load Config - Success\033[0m")

    # fault_detection_for_1_point()
    # print("\033[43mTest: Fault Detection for 1 point - Success\033[0m")

    # fault_detection_with_config_for_1_point()
    # print("\033[43mTest: Fault Detection With Config for 1 point - Success\033[0m")

    # fault_detection_multiple_variable_with_config_for_1_point()
    # print("\033[43mTest: Fault Detection Multiple Variables With Config for 1 point - Success\033[0m")

    pr = cProfile.Profile()
    pr.enable() # 프로파일링 시작
    fault_detection_multiple_variable_with_config_for_several_pointsV1()
    print("\033[43mTest: Fault Detection Multiple Variables With Config for Several pointsV1 - Success\033[0m")
    pr.disable() # 프로파일링 종료

    # 결과를 문자열 버퍼로 리다이렉션합니다.
    result_buffer = StringIO()
    stats = pstats.Stats(pr, stream=result_buffer).sort_stats('cumulative')

    # 원하는 개수의 결과만 출력합니다. 예: 상위 10개
    stats.print_stats(20)
    

    print(result_buffer.getvalue())

    pr = cProfile.Profile()
    pr.enable() # 프로파일링 시작
    fault_detection_multiple_variable_with_config_for_several_pointsV2()
    print("\033[43mTest: Fault Detection Multiple Variables With Config for Several pointsV2 - Success\033[0m")
    pr.disable() # 프로파일링 종료

    # 결과를 문자열 버퍼로 리다이렉션합니다.
    result_buffer = StringIO()
    stats = pstats.Stats(pr, stream=result_buffer).sort_stats('cumulative')

    # 원하는 개수의 결과만 출력합니다. 예: 상위 10개
    stats.print_stats(20)
    

    print(result_buffer.getvalue())