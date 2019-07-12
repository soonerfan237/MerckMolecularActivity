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
            for j in range(len(activity)):
                if features[1][i] == activity[j]:
                    features[1][i] = normalized_activity[j]

    return molecule_dict