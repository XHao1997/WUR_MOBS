import numpy as np


def slide_win(grass_list, num, threshold):
    flag = False
    grow_list = []
    harvest_to_mass = None
    go_final_harvest = False
    if len(grass_list) > num:
        for i in range(len(grass_list) - num):
            grow_rate = (grass_list[i + num] - grass_list[i]) / num
            grow_list.append(grow_rate)
            if grow_rate < 0:
                go_final_harvest = True
        if len(grow_list) > 2:
            if abs(grow_list[-1] - np.max(grow_list)) > threshold:
                flag = True
            harvest_to_mass = grass_list[np.argmax(grow_list)]
    return flag, harvest_to_mass, go_final_harvest
