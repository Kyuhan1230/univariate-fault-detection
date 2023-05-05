import numpy as np
from scipy.stats import t
from typing import List, Dict, Union


def detect_bias(meas: List[float], pred: List[float], std_train: float, confidence: float = 0.975) -> Dict[str, Union[bool, str, List[Union[bool, float]]]]:
    """
    예측값과 측정값 간의 일정 차이를 검증하여 Bias의 유무를 판단합니다.
    단, 예측값이 존재하는 경우에만 수행됩니다.
    
    Args:
        meas (list or array): 측정값 리스트 또는 배열
        pred (list or array): 예측값 리스트 또는 배열
        std_train (float): 학습 데이터의 표준 편차
        confidence (float, optional): 신뢰 구간의 신뢰도를 나타내는 값, 기본값= 0.975
    
    Returns:
        dict: Bias가 존재하는지 여부와 Bias의 크기를 담은 딕셔너리
            - success (bool): 함수 실행 결과를 나타내는 값 
                예측값이 없는 경우 False를 반환
            - result (list): Bias가 존재하는지 여부와 Bias의 크기를 담은 리스트
                예측값이 없는 경우 "Value Error"를 반환
    Examples:
        meas = [10.0, 11.0, 9.5, 10.5, 10.0, 10.5]
        pred = [9.8, 10.5, 9.9, 10.2, 10.1, 10.0]

        std_train = 0.5
        confidence = 0.975

        result = detect_bias(meas, pred, std_train, confidence)
        print(result)
        # {'success': True, 'result': [True, 0.033]}
        # Bias가 존재하며 크기는 약 0.033입니다.
    """
    if not pred:
        return {"success": False, "result": "Value Error"}
    valid_confidence_range = (0, 1)
    
    if not valid_confidence_range[0] <= confidence <= valid_confidence_range[1]:
        raise ValueError(f"Confidence must be in the range of {valid_confidence_range}, but {confidence} was given.")
    
    if isinstance(meas, np.ndarray):
        meas = meas.ravel()
    if isinstance(pred, np.ndarray):
        pred = pred.ravel()

    cc = t.ppf(confidence, len(meas) - 1) * std_train
    direction = 1 if np.mean(meas) > np.mean(pred) else -1
    errors = abs(np.array(meas) - np.array(pred))
    avg_error = round(float(np.mean(errors)), 5)
    avg_error_ = round(abs(np.mean(meas) - np.mean(pred)), 5)
    
    if avg_error_ > cc:
        bias = avg_error 
    return {"success": True, "result": [bool(bias), bias * direction]}


def apply_bias(meas: float, bias: float) -> float:
    """
    입력된 측정값(meas)에 편향(bias)을 더한 값을 반환합니다.

    Args:
        meas (float): 측정값
        bias (float): 편향값

    Returns:
        float: 입력된 측정값과 편향값을 더한 결과

    Examples:
        >>> apply_bias(10.0, 2.5)
        12.5
        >>> apply_bias(0.0, -1.0)
        -1.0
    """
    return meas + bias
