#2nd Kaggle competition in R this time.
library(corrplot)
library(MASS)

#TODO
#clean data - remove lots of NA values. 
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

#to be continued

#Let's begin with stepwise forward regression
model = lm(SalePrice ~ ., data=trainData)
#returns an error due to one category having a singleton value. Let's find it.
table(trainData$SaleType)
table(trainData$SaleCondition)
table(trainData$PoolQC) #<- no data at all, removing.
table(trainData$PavedDrive)
table(trainData$GarageCond)
table(trainData$GarageQual)
table(trainData$GarageFinish)
table(trainData$GarageType)
table(trainData$FireplaceQu)
table(trainData$Alley) #NA here means No alley access, not missing data. Need to recode.
trainData$Alley[is.na(trainData$Alley)]="No"
table(trainData$Fence) #sparse
table(trainData$MiscFeature) #sparse and not all classes
table(trainData$Utilities) #this has a single NoSeWA class. Let's remove and see what happens.
table(trainData$Condition2) #This, RoofMatl, and Condition 1 have very sparse classes. 

vectorOfSingleton=list()
for (i in 1:length(trainData)){
  vectorOfSingleton[i] = length(unique(trainData[,i]))
}

test = trainData[-5]
#So no actual issues.

trainData=trainData[,names(trainData)!="Utilities"]
testData=testData[,names(testData)!="Utilities"]

#Issue ended up being the NA values in PoolQC, Fence, and MiscFeature. These have so few values I feel comfortable removing them.
#Need to do some more data cleaning for NA values before actually doing stepwise selection. 
trainData=trainData[,names(trainData)!="PoolQc"]
testData=testData[,names(testData)!="PoolQc"]

trainData=trainData[,names(trainData)!="Fence"]
testData=testData[,names(testData)!="Fence"]

trainData=trainData[,names(trainData)!="MiscFeature"]
testData=testData[,names(testData)!="MiscFeature"]

#NA probably has something to do with the lm ~ . not working, until then this is the full model.
model <- lm(SalePrice ~ MSSubClass+MSZoning+LotFrontage+LotArea+Street+Alley+LotShape+LandContour+LandSlope
            +Neighborhood+Condition1+Condition2+BldgType+HouseStyle+OverallQual+OverallCond+YearBuilt+YearRemodAdd
            +RoofStyle+RoofMatl+Exterior1st+Exterior2nd+MasVnrType+MasVnrArea+ExterQual+ExterCond+Foundation+BsmtQual
            +BsmtCond+BsmtExposure+BsmtFinType1+BsmtFinSF1+BsmtFinType2+BsmtFinSF2+BsmtUnfSF+TotalBsmtSF+Heating
            +HeatingQC+CentralAir+Electrical+X1stFlrSF+X2ndFlrSF+LowQualFinSF+GrLivArea+BsmtFullBath+BsmtHalfBath
            +FullBath+HalfBath+BedroomAbvGr+KitchenAbvGr+KitchenQual+TotRmsAbvGrd+Functional+Fireplaces+FireplaceQu
            +GarageType+GarageYrBlt+GarageFinish+GarageCars+GarageArea+GarageQual+GarageCond+PavedDrive+WoodDeckSF
            +OpenPorchSF+EnclosedPorch+X3SsnPorch+ScreenPorch+PoolArea+MiscVal+MoSold+YrSold+SaleType+SaleCondition,data=trainData)

smallModel <- lm(SalePrice ~ YrSold+SaleType,data=trainData)