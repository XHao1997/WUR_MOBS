import numpy as np


def slide_win(grass_list, num, threshold):
    flag = False
    grow_list = []
    harvest_mass = None
    go_final_harvest = False
    harvest_day = 0
    if len(grass_list) > num:
        for i in range(len(grass_list) - num):
            grow_rate = (grass_list[i + num] - grass_list[i]) / num
            grow_list.append(grow_rate)
            if grow_rate < 0:
                go_final_harvest = True
        if len(grow_list) >= 2:
            if np.max(grow_list) - grow_list[-1] > threshold and grow_list[-1]>0:
                flag = True
                # print([len(grass_list)])
                # print(len(grow_list))
                # print('max position:', np.argmax(grow_list)+num)
                # print('max value:', grass_list[np.argmax(grow_list)+num])
                harvest_to_mass = max(grass_list[np.argmax(grow_list)] *0.5, 0.1)
                harvest_mass = grass_list[-1] - harvest_to_mass
                harvest_day = np.argmax(grow_list) + num
    return flag, harvest_mass, go_final_harvest, harvest_day
