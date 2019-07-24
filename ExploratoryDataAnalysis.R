library(Amelia) #needed for missmap
library(ggplot2)  #for graphics
library(gridExtra) #for arranging graphs

setwd("/Users/soonerfan237/Desktop/MerckActivity/TrainingSet15")
print(getwd())

molecule_dict <- read.csv("molecule_dict_filter.csv", stringsAsFactors=FALSE)  #reading in exoplanet data
#summary(molecule_dict)

#CORRELATION
#correlation <- cor(molecule_dict$ACT4, molecule_dict)
#test_filter <- correlation[correlation>0.2]
#test_filter
histogram.ACT1 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT1)) + geom_density() + xlab("feature1") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT2 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT2)) + geom_density() + xlab("feature2") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT3 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT3)) + geom_density() + xlab("feature3") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT4 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT4)) + geom_density() + xlab("feature4") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT5 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT5)) + geom_density() + xlab("feature5") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT6 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT6)) + geom_density() + xlab("feature6") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT7 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT7)) + geom_density() + xlab("feature7") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT8 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT8)) + geom_density() + xlab("feature8") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT9 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT9)) + geom_density() + xlab("feature9") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT10 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT10)) + geom_density() + xlab("feature10") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT11 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT11)) + geom_density() + xlab("feature11") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT12 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT12)) + geom_density() + xlab("feature12") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT13 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT13)) + geom_density() + xlab("feature13") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT14 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT14)) + geom_density() + xlab("feature14") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
histogram.ACT15 <- ggplot(molecule_dict,aes(x = molecule_dict$ACT15)) + geom_density() + xlab("feature15") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
#grid.arrange(histogram.ACT1, histogram.ACT2, histogram.ACT3, histogram.ACT4, histogram.ACT5, histogram.ACT6, histogram.ACT7, histogram.ACT8, histogram.ACT9, histogram.ACT10, histogram.ACT11, histogram.ACT12, histogram.ACT13, histogram.ACT14, histogram.ACT15, nrow = 5)
histogram.ACT15

#histogram.feature <- ggplot(molecule_dict,aes(x = molecule_dict$D_62)) + geom_density() + xlab("feature") + scale_color_gradient(low="blue", high="red") + theme(legend.position="bottom") 
#histogram.feature
#MISSINGNESS
#missmap(molecule_dict) #missingness map

#TODO: look at the distribution of acitivites just within a few features to get an idea for the range and distribution
#can features be negative?
#does 0 mean not present?
#how does this inform missing value imputation