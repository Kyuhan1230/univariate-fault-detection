def get_matching_keys(res):
    """
    주어진 결과 딕셔너리에서 success와 fault_detected가 True인 경우에만,
    각각 Frozen, Boundary, Dynamics, Drift 키의 값을 확인하여
    True인 경우에 해당하는 키를 반환하는 함수입니다.

    Args:
        res (dict): 결과를 포함하는 딕셔너리. 다음 키를 포함해야 합니다:
                   'success', 'fault_detected', 'Frozen', 'Boundary', 'Dynamics', 'Drift', 'values'.

    Returns:
        list or None: 조건에 맞는 경우, 각 키에 해당하는 키 값을 포함하는 리스트를 반환하고,
                      그렇지 않으면 None을 반환합니다.

    Examples:
        res = {'success': True,
               'fault_detected': True,
               'Frozen': True,
               'Boundary': False,
               'Dynamics': False,
               'Drift': False,
               'values': {'frozen': 0.84, 'boundary': [False, 1190.98], 'dynamics': 0.84, 'drift': [0, 0]}}

        result = get_matching_keys(res)
        print(result)  # ['Frozen']
    """
    if res['success'] and res['fault_detected']:
        result = []
        keys = ['Frozen', 'Boundary', 'Dynamics', 'Drift']
        for key in keys:
            if res[key]:
                result.append(key.lower())
        return result
    return None