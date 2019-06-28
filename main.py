import NormalizeData
import FindFeatures
import NeuralNet

def main():

    data_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSUBSet2/"

    NormalizeData.NormalizeData(data_directory)

    FindFeatures.FindFeatures(data_directory)

    for i in [4,5]:
        NeuralNet.NeuralNet(data_directory,i) #integer corresponds to activity number to predict

main()