import numpy as np
from sklearn.preprocessing import minmax_scale

def NormalizeActivity(molecule_dict, activity_list):

    activity = []

    for i in activity_list:
        for molecule, features in molecule_dict.items():
            activity.append(features[1][i])

    activity = np.array(activity)
    normalized_activity = minmax_scale(activity)

    for i in activity_list:
        for molecule, features in molecule_dict.items():
            j=0
            while features[1][i] != activity[j] and j < len(activity):
                j+=1
            features[1][i] = normalized_activity[j]

    return molecule_dict