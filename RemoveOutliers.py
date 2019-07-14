import numpy as np
import copy #needed to deepcopy dictionary so i can delete stuff during loop

def RemoveOutliers(molecule_dict_filter):
    print("STARTING RemoveOutliers")
    print("TOTAL MOLECULES: " + str(len(molecule_dict_filter)))

    feature_totals = []
    for molecule, values in molecule_dict_filter.items():
        total = float(0)
        for value in values[0]:
            total = total + float(value)
        feature_totals.append(total)

    feature_totals = np.array(feature_totals)
    feature_totals = feature_totals.astype(float)
    stdev = np.std(feature_totals)
    mean = np.mean(feature_totals)

    molecule_dict_filter_new = copy.deepcopy(molecule_dict_filter)
    delete_count = 0
    for molecule, values in molecule_dict_filter.items():
        total = float(0)
        for value in values[0]:
            total = total + float(value)
        if abs(total-mean) > 2*stdev:
            del molecule_dict_filter_new[molecule]
            delete_count += 1

    print("REMOVED OUTLIERS: " + str(delete_count))
    print("REMAINING MOLECULES: " + str(len(molecule_dict_filter_new)))

    return molecule_dict_filter_new