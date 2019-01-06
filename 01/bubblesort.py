import numpy as np


def bubblesort(list) -> (np.array):
    """
    Implement Bubblesort using an np.array as input and
    returning a sorted np.array

    :param list:np.array
    :return: np.array
    """
    # pass does not do anything but is necessary to run this code
    for i in range(len(list), 1, -1):
        for j in range(i - 1):
            if list[j] > list[i]:
                list[[i, i + 1]] = list[[i + 1, i]]
    return list


if __name__ == '__main__':
    # use np.arange to create an numpy array and
    # shuffle all values using np.random.shuffle
    # please read the document for these functions
    pass
    a = np.random.randint(0, 50, 20)
    b = bubblesort(a)
    print(a)
    print(b)
