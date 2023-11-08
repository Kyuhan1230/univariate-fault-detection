
import pandas as pd
import numpy as np


DATA_TYPES = (np.ndarray, pd.DataFrame)
TAG_TYPES = (list, tuple)


def all_elements_equal(lst):
    return all(x == lst[0] for x in lst)


def validate_data(data, tag_list):
    if not isinstance(data, DATA_TYPES):
        raise TypeError("Data Should be Numpy ndarray or DataFrame")
    if isinstance(data, pd.DataFrame) and tag_list is None:
        tag_list = data.columns.to_list()
        data = data.values
    if not isinstance(tag_list, TAG_TYPES):
        raise TypeError("tag list should be list. ex) tag_list=['tag01', 'tag2', ..., 'tag10']")
    if len(tag_list) != data.shape[1]:
        raise ValueError("The number of variables in the input data and the number of variables in the tag list must match.")
    return data, tag_list
