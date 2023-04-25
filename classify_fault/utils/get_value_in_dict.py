def get_value(dictionary:dict, key, default_value):
    """
    딕셔너리에서 key의 값을 반환하며, key가 존재하지 않을 경우 default_value를 반환합니다.
    """
    return dictionary.get(key, default_value)