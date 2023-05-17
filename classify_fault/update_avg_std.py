from math import sqrt


def update_avg_std(avg_old, std_old, new_value, update_option, oldest_value, data_size):
    """Update statistics (average, standard deviation) based on a new value.

    Args:
        avg_old (float): The current average of the dataset.
        std_old (float): The current standard deviation of the dataset.
        new_value (float): The new value to be added to the dataset.
        update_option (str): The update option to be used. Valid options are "Keep Size" or "Increase".
        oldest_value (float): The oldest value in the dataset, to be used when using the "Keep Size" update option.
        data_size (int): The size of the dataset.

    Returns:
        tuple: A tuple containing the updated average and standard deviation.

    Raises:
        TypeError: If the update option is not a string or is not one of the valid options.
    """
    if isinstance(update_option, str):
        option = update_option.lower()
        if 'keep' in option:
            option = 'keep'
            if oldest_value is None:
                raise TypeError("Keep Size Option should has Initial Value.")
        else:
            option = 'increase'

    else:
        raise TypeError("Option should be string, one of 'Keep Size', 'Increase' ")

    if option == 'increase':
        avg_updated = update_average_incremental(average=avg_old,
                                                 new_value=new_value,
                                                 data_size=data_size)
        std_updated = update_std_incremental(std_old=std_old, avg_old=avg_old,
                                             avg_new=avg_updated, new_value=new_value,
                                             data_size=data_size)
    else:
        avg_updated = update_average_sliding(average=avg_old,
                                             new_value=new_value,
                                             oldest_value=oldest_value,
                                             data_size=data_size)
        std_updated = update_std_sliding(std_old=std_old, avg_old=avg_old,
                                         avg_new=avg_updated, new_value=new_value,
                                         oldest_value=oldest_value,
                                         data_size=data_size)

    return avg_updated, std_updated


def update_average_sliding(average, new_value, oldest_value, data_size):
    """Update the average while keeping the data size constant.

    Args:
        average (float): The current average of the dataset.
        new_value (float): The new value to be added to the dataset.
        oldest_value (float): The oldest value in the dataset.
        data_size (int): The size of the dataset.

    Returns:
        float: The updated average.

    """
    return float((data_size * average - oldest_value + new_value) / data_size)


def update_average_incremental(average, new_value, data_size):
    """Update the average after adding a new value to the data

    Args:
        average (float): The current average of the dataset.
        new_value (float): The new value to be added to the dataset.
        data_size (int): The size of the dataset.

    Returns:
        float: The updated average.

    """
    return float((data_size * average + new_value) / (data_size + 1))


def update_std_sliding(std_old, avg_old, avg_new, new_value, oldest_value, data_size):
    """Update the standard deviation while keeping the data size constant.

    Args:
        std_old (float): The current standard deviation of the dataset.
        avg_old (float): The current average of the dataset.
        avg_new (float): The new average of the dataset.
        new_value (float): The new value to be added to the dataset.
        oldest_value (float): The oldest value in the dataset.
        data_size (int): the size of the dataset

    Returns:
        float: The updated standard deviation.
        reference: https://nestedsoftware.com/2019/09/26/incremental-average-and-standard-deviation-with-sliding-window-470k.176143.html
    """
    if data_size == 0:
        raise ValueError("data_size cannot be zero")
    inside_sqrt = (
        std_old ** 2 + ((new_value - oldest_value) * (new_value + oldest_value - avg_new - avg_old)) / data_size
    )

    if inside_sqrt < 0:
        return std_old

    return float(sqrt(inside_sqrt))


def update_std_incremental(std_old, avg_old, avg_new, new_value, data_size):
    """Update the standard deviation after adding a new value to the data

    Args:
        std_old (float): the previous standard deviation of the dataset
        avg_old (float): the previous average of the dataset
        avg_new (float): the new average of the dataset
        new_value (float): the new value that is being added to the dataset
        data_size (int): the size of the dataset

    Returns:
        float: The updated standard deviation.
        reference: https://www.physicsforums.com/threads/updating-the-mean-and-sd-of-a-set-efficiently.526280/
    """
    return float(sqrt((std_old ** 2 * data_size + (new_value - avg_new) * (new_value - avg_old)) / (data_size + 1)))
