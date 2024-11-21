import time

#printSeries
def print_cust(info, color:str='black'):
    if isinstance(info, str):
        head = info[:7].lower()
        if head == 'success':
            color = 'green'
        elif head == 'warning':
            color = 'red'
    end = '\033[0m'
    colors = {'green':'\033[32m', 'black':'\033[30m', 'red':'\033[31m', 'yellow':'\033[33m', \
              'blue': '\033[34m', 'white':'\033[37m'}
    try:
        assert color in colors.keys()
    except:
        print(f"{colors['red']}Warning: {color} is not in the list, use blue as default.{end}", end='\n')
    print(f"{colors[color]}{info}{end}")


# Decoration
def timeit(func):
    '''
    Decoration function to count time consuming.
    '''
    def wrap(*args, **kwargs):
        t_0 = time.time()
        result = func(*args, **kwargs)
        print_cust("Time Computation Info:\tThe '{}' Function takes {:<5.2f}s.".format(func.__name__, time.time()-t_0), 'yellow')
        return result
    return wrap


def normalize(tensor, mean, std):
    '''
    Normalize the tensor with mean and std.
    Only support 1D, 2D, 3D, 4D tensor / ndarray, list is not acceptable.
    '''
    if len(mean) != len(std):
        raise ValueError(f"The length of mean and std should be the same, but got {len(mean)} and {len(std)}")

    if tensor.dim() == 4:  # [b, c, h, w] format
        for i in range(len(mean)):
            tensor[:, i, :, :] = (tensor[:, i, :, :] - mean[i]) / std[i]
    elif tensor.dim() == 3:  # [c, h, w] format
        for i in range(len(mean)):
            tensor[i, :, :] = (tensor[i, :, :] - mean[i]) / std[i]
    elif tensor.dim() == 2:  # 2D array
        for i in range(len(mean)):
            tensor[:, i] = (tensor[:, i] - mean[i]) / std[i]
    elif tensor.dim() == 1:  # 1D array
        for i in range(len(mean)):
            tensor[i] = (tensor[i] - mean[i]) / std[i]
    else:
        raise ValueError("Unsupported tensor shape")

    return tensor


def normalize_01(tensor, max_val, min_val):
    return (tensor - min_val) / (max_val - min_val)