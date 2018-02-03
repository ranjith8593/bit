import concurrent.futures
import os
import pandas as pd

def simple_file_read(file_name):
    f = open(file_name, 'r')
    for a in f:
        yield eval(a)

def get_data_frame(file):
    if not os.path.exists(file):
        print("No file named", file)
    i = 0
    dict = {}
    for line in simple_file_read(file):
        dict[i] = line
        i+=1
    data_frame =pd.DataFrame.from_dict(dict, orient="index")
    print("data loaded", data_frame.shape)
    return data_frame


def parallel_execute(function, list_values, workers=4):
    result_array =[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for i, result in enumerate(executor.map(function, list_values)):
            print("\rCompleted==>", i, end="")
            result_array.append(result)

    return result_array