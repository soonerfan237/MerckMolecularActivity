import CombineData
import NormalizeActivity
import FindVariableFeatures
import FindCorrelatedFeatures
import RemoveOutliers
import NeuralNet
import ConvNeuralNet
import GenerativeAdversarialNetwork
import glob
import re
import time

def main():
    start_time = time.time()
    data_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSet4/"
    files = glob.glob(data_directory + "ACT*.csv")
    activity_list = []
    for file in files:
        print(file)
        match = re.search(r'ACT([0-9]+)', file)
        activity_list.append(int(match.group(1)))

    feature_dict, molecule_dict = CombineData.CombineData(data_directory)
    molecule_dict = NormalizeActivity.NormalizeActivity(molecule_dict, activity_list)
    feature_dict_filter, molecule_dict_filter = FindVariableFeatures.FindVariableFeatures(data_directory, feature_dict, molecule_dict)
    #molecule_dict_filter = FindCorrelatedFeatures.FindCorrelatedFeatures(feature_dict_filter, molecule_dict_filter, activity_list)
    #molecule_dict_filter = RemoveOutliers.RemoveOutliers(molecule_dict_filter)

    for i in activity_list:
        #NeuralNet.NeuralNet(data_directory, i, molecule_dict_filter) #i corresponds to activity number to predict
        #ConvNeuralNet.ConvNeuralNet(data_directory, i, molecule_dict_filter)
        GenerativeAdversarialNetwork.GenerativeAdversarialNetwork(data_directory, i, molecule_dict_filter)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    print("DONE!")
main()