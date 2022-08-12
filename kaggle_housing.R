#2nd Kaggle competition in R this time.
library(corrplot)
library(MASS)
library(mice)
#TODO
#clean data - remove lots of NA values. We will need to recode the categorical data to numeric (0,1,2, etc.), decide what to drop, and what to impute 
#and how to impute it!
#EDA
#begin model selection
#figure out important variables
#get another model
#finalize it
testData = read.csv("C:\\Users\\BSeng\\OneDrive - Minitab, LLC\\Documents\\house-prices-advanced-regression-techniques\\test.csv",header=TRUE)
trainData = read.csv("C:\\Users\\BSeng\\OneDrive - Minitab, LLC\\Documents\\house-prices-advanced-regression-techniques\\train.csv",header=TRUE)
trainDataID = trainData$Id
trainData=trainData[-1] #drop ID value
testDataID = testData$Id
testData=testData[-1] #drop ID value
#EDA

summary(trainData) #variables are quite different from each other. Some categorical, some continuous. 
#LotFrontage has many missings. GarageYrBlt and MasVnrArea also have missings. Optional house features, like BsmtFin, will be skewed.
str(trainData)
#Issue ended up being the NA values in PoolQC, Fence, and MiscFeature. These have so few values I feel comfortable removing them.
colSums(is.na(trainData)) #as confirmed, lotFrontage, PoolQC, FireplaceQu, miscFeature,Fence. Most I'll remove, some I want to keep.

trainData=trainData[,names(trainData)!="Alley"]
testData=testData[,names(testData)!="Alley"]

trainData=trainData[,names(trainData)!="PoolQc"]
testData=testData[,names(testData)!="PoolQc"]

trainData=trainData[,names(trainData)!="Fence"]
testData=testData[,names(testData)!="Fence"]

trainData=trainData[,names(trainData)!="MiscFeature"]
testData=testData[,names(testData)!="MiscFeature"]

trainData=trainData[,names(trainData)!="FireplaceQu"]
testData=testData[,names(testData)!="FireplaceQu"]

trainData=trainData[,names(trainData)!="Utilities"]
testData=testData[,names(testData)!="Utilities"]

#Let's begin with the categorical data.
cateDat=which(sapply(trainData,is.character)==TRUE)

table(trainData$MSZoning)
#RH, RM, RL are all residential and the others are not. Let's have 3 categories. We could also have NAs as 0 with a binary column to denote missings...
trainData$temp[trainData$MSZoning %in% c("FV")] <- 3
trainData$temp[trainData$MSZoning %in% c("RH","RM","RL")] <- 2
trainData$temp[trainData$MSZoning %in% c("C (all)")] <- 1
trainData$MSZoning = trainData$temp 

table(trainData$Street)
trainData$temp[trainData$Street %in% c("Grvl")] <- 2
trainData$temp[trainData$Street %in% c("Pave")] <- 1
trainData$Street = trainData$temp

table(trainData$LotShape)
trainData$temp[trainData$LotShape %in% c("IR1")] <- 4
trainData$temp[trainData$LotShape %in% c("IR2")] <- 3
trainData$temp[trainData$LotShape %in% c("IR3")] <- 2
trainData$temp[trainData$LotShape %in% c("Reg")] <- 1
trainData$LotShape = trainData$temp 

table(trainData$LandContour)
trainData$temp[trainData$LandContour %in% c("Bnk")] <- 4
trainData$temp[trainData$LandContour %in% c("HLS")] <- 3
trainData$temp[trainData$LandContour %in% c("Low")] <- 2
trainData$temp[trainData$LandContour %in% c("Lvl")] <- 1
trainData$LandContour = trainData$temp 

table(trainData$LotConfig)
#According to https://journal.firsttuesday.us/type-of-lots/70394/, it makes sense to group the frontage+corners, and cul de dac with inside.
trainData$temp[trainData$LotConfig %in% c("Corner","FR2","FR3")] <- 2
trainData$temp[trainData$LotConfig %in% c("CulDSac","Inside")] <- 1
trainData$LotConfig = trainData$temp 



#now we'll look at the continuous data. Here we might just impute based on means for now. 
contDat=which(sapply(trainData,is.numeric)==TRUE)






corrplot(cor(trainData[,c(3,4,16,17,18,19)],use="everything"),sig.level = 0.01,insig="blank") 
trainData$LotFrontage[is.na(trainData$LotFrontage)] = 0 #lot frontage has NAs
#############################################





