import NormalizeData
import FindFeatures
import NeuralNet

def main():

    data_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSUBSet2/"

    feature_dict, molecule_dict = NormalizeData.NormalizeData(data_directory)

    molecule_dict_filter = FindFeatures.FindFeatures(data_directory, feature_dict, molecule_dict)

    #for i in range(1,16):
    for i in [4,5]:
        NeuralNet.NeuralNet(data_directory, i, molecule_dict_filter) #i corresponds to activity number to predict

main()