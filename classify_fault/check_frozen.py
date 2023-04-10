import numpy as np


def get_frozen_test_result(frozen_result_detect: dict, tag_name: str) -> dict:
    """
    Frozen Test: 데이터 검증(Data Validation) 단계에서 수행한 Frozen Test의 결과를 반환합니다.
    
    Args:
        frozen_result_detect (dict): Frozen Test 결과를 저장한 딕셔너리
        tag_name (str): 반환할 태그의 이름
    
    Returns:
        dict: Frozen Test 결과를 담은 딕셔너리
            - success (bool): 함수 실행 결과를 나타내는 값
                frozen_result_detect에 tag_name 키가 없으면 False를 반환
                그 외에는 True를 반환
            - result (any): Frozen Test 결과 값을 담은 변수
                frozen_result_detect에 tag_name 키가 없으면 None을 반환
                그 외에는 tag_name에 해당하는 값이 반환
    
    Raises:
        ValueError: frozen_result_detect가 딕셔너리 타입이 아닌 경우 예외를 발생
    """
    if not isinstance(frozen_result_detect, dict):
        raise ValueError("frozen_result_detect should be a dictionary.")
    
    if tag_name not in frozen_result_detect:
        return {"success": False, "result": None}
    else:
        return {"success": True, "result": frozen_result_detect[tag_name]}



def detect_frozen(data, frozen_threshold=0.01, n=5):
    """
    Check for frozen states in the data based on the specified threshold.
    
    Args:
        data (numpy.ndarray): 과거 데이터의 1D array
        frozen_threshold (float): frozen state를 감지할 threshold
        n (int): 관찰할 이전 데이터의 수
    
    Returns:
        bool: frozen state를 검출한 경우 True를, 검출하지 못한 경우 False를 반환

    Examples:
        data = np.array([1, 2, 2, 2, 2, 2])
        frozen_threshold = 0.5
        n = 3

        frozen_detected = detect_frozen(data, frozen_threshold, n)
        print(f"Is frozen state detected? {frozen_detected}")
        # Is frozen state detected? True
    """
    diff = np.abs(np.diff(data[-n:]))    # Calculate the absolute difference between recent N data points
    avg_diff = np.mean(diff)

    if avg_diff <= frozen_threshold:
        return True
    return False

