import CombineData
import NormalizeActivity
import FindVariableFeatures
import FindCorrelatedFeatures
import NeuralNet
import ConvNeuralNet
import glob
import re

def main():

    data_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSet8/"
    files = glob.glob(data_directory + "ACT*.csv")
    activity_list = []
    for file in files:
        print(file)
        match = re.search(r'ACT([0-9]+)', file)
        activity_list.append(int(match.group(1)))

    feature_dict, molecule_dict = CombineData.CombineData(data_directory)
    molecule_dict = NormalizeActivity.NormalizeActivity(molecule_dict)
    feature_dict_filter, molecule_dict_filter = FindVariableFeatures.FindVariableFeatures(data_directory, feature_dict, molecule_dict)
    molecule_dict_filter = FindCorrelatedFeatures.FindCorrelatedFeatures(feature_dict_filter, molecule_dict_filter, activity_list)

    for i in activity_list:
        NeuralNet.NeuralNet(data_directory, i, molecule_dict_filter) #i corresponds to activity number to predict
        #ConvNeuralNet.ConvNeuralNet(data_directory, i, molecule_dict_filter)

main()