import numpy as np
from sklearn.preprocessing import minmax_scale

def NormalizeActivity(molecule_dict, activity_list):

    print("STARTING NormalizeActivity")

    activity = []

    for i in activity_list:
        for molecule, features in molecule_dict.items():
            activity.append(features[1][i])

    normalized_activity = np.array(activity)
    normalized_activity = normalized_activity.astype(float)
    #normalized_activity = normalized_activity / np.sqrt(np.sum(normalized_activity ** 2))
    normalized_activity = minmax_scale(normalized_activity)*10

    for i in activity_list:
        for molecule, features in molecule_dict.items():
            j=0
            while features[1][i] != activity[j] and j < len(activity):
                j+=1
            features[1][i] = normalized_activity[j]

    print("DONE!")
    return molecule_dict