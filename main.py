import NormalizeData
import FindFeatures
import NeuralNet

def main():

    data_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSet/"

    NormalizeData.NormalizeData(data_directory)

    FindFeatures.FindFeatures(data_directory)

    for i in range(1,16):
        NeuralNet.NeuralNet(data_directory,i) #integer corresponds to activity number to predict

main()