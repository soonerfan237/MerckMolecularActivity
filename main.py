import NormalizeData
import FindFeatures
import NeuralNet
import ConvNeuralNet
import glob
import re

def main():

    data_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSet4/"
    files = glob.glob(data_directory + "ACT*.csv")
    activity_list = []
    for file in files:
        print(file)
        match = re.search(r'ACT([0-9]+)', file)
        activity_list.append(int(match.group(1)))

    feature_dict, molecule_dict = NormalizeData.NormalizeData(data_directory)

    molecule_dict_filter = FindFeatures.FindFeatures(data_directory, feature_dict, molecule_dict)

    for i in activity_list:
        #NeuralNet.NeuralNet(data_directory, i, molecule_dict_filter) #i corresponds to activity number to predict
        ConvNeuralNet.ConvNeuralNet(data_directory, i, molecule_dict_filter)

main()