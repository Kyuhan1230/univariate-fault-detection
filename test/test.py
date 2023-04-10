import os
import sys
import numpy as np

sys.path.append(os.path.abspath('./'))

from classify_fault.set_config import calculate_variables_config, save_config, load_config
from classify_fault.fault_detection import detect_fault


def test_fault_detection_with_config():
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

    save_config(variables_config, "./config/test_config.yaml")
    loaded_config = load_config("./config/test_config.yaml")

    data_to_test = np.array([[10, 20]])

    result = detect_fault(data_to_test[0], 
                          loaded_config["Var1"]["tracking_size"], loaded_config["Var1"]["type_to_check"],
                          loaded_config["Var1"]["frozen_threshold"], loaded_config["Var1"]["boundary_limits"],
                          loaded_config["Var1"]["dynamic_threshold"], loaded_config["Var1"]["drift_params"])


    assert result["success"], f"Error occurred: {result['message']}"
    assert result["fault_detected"]
    assert result["Frozen"]==False
    assert result["Boundary"]
    assert result["Dynamics"]
    assert result["Drift"]==False
    # 파일 삭제
    os.remove("./config/test_config.yaml")


def test_fault_detection():
    data = np.array([1, 2, 2, 2, 2, 10])
    type_to_check = {
        "frozen": True,
        "boundary": True,
        "dynamics": True,
        "drift": True
    }
    frozen_threshold = 0.5
    tracking_size = 3
    boundary_limits = (4, 0)
    dynamic_threshold = 1
    drift_params = {"average": 2, "cusum_threshold": 4.0, "ewma_alpha": 0.2}

    result = detect_fault(data, tracking_size, type_to_check, frozen_threshold, boundary_limits, dynamic_threshold, drift_params)

    assert result["success"] == True, f"Error occurred: {result['message']}"
    assert result["fault_detected"] == True, f"Error occurred: {result}"
    assert result["Frozen"] == False, f"Error occurred: {result}"
    assert result["Boundary"] == True, f"Error occurred: {result}"
    assert result["Dynamics"] == True, f"Error occurred: {result}"
    assert result["Drift"] == False, f"Error occurred: {result}"


def test_save_load_config():
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

    save_config(variables_config, "./config/test_config.yaml")
    loaded_config = load_config("./config/test_config.yaml")

    assert variables_config == loaded_config
    # 파일 삭제
    os.remove("./config/test_config.yaml")

if __name__ == "__main__":
    test_fault_detection_with_config()
    test_fault_detection()
    test_save_load_config()