import os
import sys
import numpy as np

sys.path.append(os.path.abspath('./'))

from classify_fault.set_config import calculate_variables_config, save_config, load_config
from classify_fault.fault_detection import detect_fault


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
    save_load_config()
    print("\033[43mTest: Save Load Config - Success\033[0m")

    fault_detection_for_1_point()
    print("\033[43mTest: Fault Detection for 1 point - Success\033[0m")

    fault_detection_with_config_for_1_point()
    print("\033[43mTest: Fault Detection With Config for 1 point - Success\033[0m")
