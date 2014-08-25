---
output: html_document
---
Predicting Form of Weight Lifting Exercises - Machine Learning Course Project
========================================================

Script:machineLearning.Rmd was compiled and tested on a PC using R Version 3.1.1.
Authored by Sarah Sayed

This document was created as a project for the course [Machine Learning](https://www.coursera.org/course/predmachlearn) Coursera course.

This script evaluates machine learning models predictions to predict one of five weight lifting exercises methodology using the Human Activity Recognition (HAR) wearable device data obtained from a study conducted by [Velloso, E. et all.](http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf)

For the study, six healthy participants aged 20-28, were asked to perform one set of 10 repetitions of unilateral dumbbell biceps Curls using five different methods. (classe variables in the dataset.)


A: exactly according to the specification
B: throwing the elbows to the front (common error)
C: lifting the dumbbell only halfway
D: lowering the dumbbell only halfway
E: throwing the hips to the front 

Note that class A represents the exercise conducted with proper technique and class B-E represent common exercise mistakes made by novices.

A prediction model was designed to predict class using the train dataset. Once the model was constructed test dataset values were used to predict classe values. Random Forest prediction model evaluated with the train() function was found to give the highest accuracy (97%).  

 
## Data Exploration

Unzip and read the 'pml-training.csv' and 'pml-testing.csv' data files.


```r
#training Data
myurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
setInternet2(use = TRUE) ## Correction for windows device
download.file(url=myurl, destfile="Training.csv")

train <- read.csv("Training.csv", header = TRUE, stringsAsFactors = F, comment.char="", na.strings="NA")
#dim(training) 19622 160

#testing Data
myurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url=myurl, destfile="Testing.csv")

test <- read.csv("Testing.csv", header = TRUE, stringsAsFactors = F, comment.char="", na.strings="NA") 
#dim(testing) 20 160
```


Out of the 160 variables in the dataset, remove unnecessary variables. The 'new window' variable is sparsely populated, furthermore this variable is not populated in the given test set data. Therefore it will be removed by subsetting off the rows of the dataset that the new_Window variable is = 'no'. 

The nearZeroVar() function will be used to remove values with no variance (all the variables related to the new_window parameter).


```r
library(caret)
library(Hmisc)

train <- subset(train, new_window == "no") 
## Remove values with where the new_widow variable = "Yes", since test set only has  new_window='no'

nzv <- nearZeroVar(train, saveMetrics = TRUE) ## remove values with 0 variance (all the variables only related to the new_Window parameter)

#where nzv is the predictor of the near zero varience
nearZeroVar <- subset(nzv, nzv == "TRUE") ## dismissed due to near zero variance
modelVariables <- subset(nzv, nzv == "FALSE") ## values to keep

## Remove all the columns that contain zero variance
train <- train[ , which(names(train) %in% rownames(modelVariables))]
```


Use the describe() function in the 'psych' package to visually inspect remaining columns and to remove columns that likely bear no relevancy in the prediction model (values that are hardly populated, or are not numeric), such as time and user name parameters.


```r
library(psych)

#describe(train)
train <- train[,-which(names(train) %in% c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "num_window"))] 
# dim(train) 19216 53
```

Now the dataset has been reduced from containing 160 variables, to 53 (52 variables to predict the classe outcome.).

Remove the same variables from the test set (identical process to training for later when we verify the model). (Note that the test dataset will not be used to construct the model, it will only be used to predict the classes at the end.)

```r
library(psych)

nzvTest <- nearZeroVar(test, saveMetrics = TRUE) ## remove values with 0 variance (all the variables only related to the new_Window parameter)

#where nzv is the predictor of the near zero varience
nearZeroVarTest <- subset(nzvTest, nzv == "TRUE") ## dismissed due to near zero variance
modelVariablesTest <- subset(nzvTest, nzv == "FALSE") ## values to keep

## Remove all the columns that contain zero variance
test <- test[ , which(names(test) %in% rownames(modelVariablesTest))]

test <- test[,-which(names(test) %in% c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "num_window"))] 

#dim(test) 20 53
```


At this point we are ready to further subset the training data into test and training data to construct a model


```r
library(AppliedPredictiveModeling)

# Further subset the training data into test and training data
inTrain <- createDataPartition(train$classe, p=0.6, list = FALSE)
training <- train[inTrain,] #dim(training) 11532 53
testing <- train[-inTrain,] #dim(testing) 7684 53

#Convert the output variable 'classe' into a factor variable
#training$classe <- as.factor(training$classe) 
#testing$classe <- as.factor(testing$classe)
```

Use the train() function to run the random forest model by setting method = 'rf' in the train function. the trControl parameter was set to reduce the time to process the function. Cross Validaton with four folds and preprocessing of the training set data (centering and scaling) was used.

Note other models, such as the 'rpart' model was also tried, however the other models contained a lower accuracy rate.

```r
library(caret)
library(Hmisc)
library(randomForest)

set.seed(125)
#Various models were tried setting the seed before each time
training2 <- training[sample(nrow(training), 5000), ] # subset of the dataset 

# Using the train function and preProcessing the data, cross validation done in trainControl
model   <- train(as.factor(classe) ~., data = training2, method = "rf", preProcess = c("center", "scale"), trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE, verboseIter = TRUE))
```

```
## + Fold1: mtry= 2 
## - Fold1: mtry= 2 
## + Fold1: mtry=27 
## - Fold1: mtry=27 
## + Fold1: mtry=52 
## - Fold1: mtry=52 
## + Fold2: mtry= 2 
## - Fold2: mtry= 2 
## + Fold2: mtry=27 
## - Fold2: mtry=27 
## + Fold2: mtry=52 
## - Fold2: mtry=52 
## + Fold3: mtry= 2 
## - Fold3: mtry= 2 
## + Fold3: mtry=27 
## - Fold3: mtry=27 
## + Fold3: mtry=52 
## - Fold3: mtry=52 
## + Fold4: mtry= 2 
## - Fold4: mtry= 2 
## + Fold4: mtry=27 
## - Fold4: mtry=27 
## + Fold4: mtry=52 
## - Fold4: mtry=52 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 27 on full training set
```

```r
#rf.model <- randomForest(classe~. data = training2, ntree = 3000, mtry = 27) 

#library(gbm) #using the Gradiant boosting algorithm
#gbm.model <- train(classe~.,training, method="gbm",verbose=FALSE, metric="Accuracy",rControl=trainControl(method="repeatedcv", number=8, repeats=4))
```


```r
model # display model accuracy 96.6%, and other parameters
```

```
## Random Forest 
## 
## 5000 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 3750, 3748, 3750, 3752 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.006        0.008   
##   30    1         1      0.002        0.003   
##   50    1         1      0.003        0.003   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
print(model$finalModel) #display model summary
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 2.4%
## Confusion matrix:
##      A   B   C   D   E class.error
## A 1426   2   2   1   2    0.004885
## B   25 933  17   4   0    0.046987
## C    1  18 827   7   0    0.030481
## D    0   2  17 786   3    0.027228
## E    0   3   7   9 908    0.020496
```

The train() function random forest model with pre-processing and cross validation  was 
found to have the highest accuracy rate. 

Finally the model can be tested using the predict() function. First the model is tested on the train dataset (testing). Finally it is tested on test to predict classe variables.
The confusion matrix summarizes the overall accuracy of the model. Note that the in sample error rate (for the training set) is less than the out-of-sample error rate (for the testing set). 


```r
trainingPred<- predict(model, training) ## test the model on the training dataset (non-subsetted)
#table(trainingPred, training$classe)
confusionMatrix(trainingPred, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3277   21    0    0    0
##          B    4 2198   30    5    8
##          C    2   10 1972   33    3
##          D    0    1   10 1847    4
##          E    0    1    0    4 2102
## 
## Overall Statistics
##                                        
##                Accuracy : 0.988        
##                  95% CI : (0.986, 0.99)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.985        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.985    0.980    0.978    0.993
## Specificity             0.997    0.995    0.995    0.998    0.999
## Pos Pred Value          0.994    0.979    0.976    0.992    0.998
## Neg Pred Value          0.999    0.996    0.996    0.996    0.998
## Prevalence              0.285    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.191    0.171    0.160    0.182
## Detection Prevalence    0.286    0.195    0.175    0.161    0.183
## Balanced Accuracy       0.998    0.990    0.988    0.988    0.996
```

```r
trainPred <- predict(model, testing) ## test the model on the testing set (split from the train dataset)
#table(trainPred, testing$classe) ##Observe predicted values
confusionMatrix(trainPred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2179   33    0    0    0
##          B    3 1427   31    7    3
##          C    3   13 1299   43    6
##          D    2   11   10 1205    2
##          E    1    3    0    3 1400
## 
## Overall Statistics
##                                         
##                Accuracy : 0.977         
##                  95% CI : (0.974, 0.981)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.971         
##  Mcnemar's Test P-Value : 2.61e-10      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.960    0.969    0.958    0.992
## Specificity             0.994    0.993    0.990    0.996    0.999
## Pos Pred Value          0.985    0.970    0.952    0.980    0.995
## Neg Pred Value          0.998    0.990    0.994    0.992    0.998
## Prevalence              0.285    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.186    0.169    0.157    0.182
## Detection Prevalence    0.288    0.191    0.178    0.160    0.183
## Balanced Accuracy       0.995    0.976    0.980    0.977    0.996
```

```r
finalTestPred<- predict(model, test) ## Finally test the model on test dataset values
```


Using the model above, the predicted output may be printed using the function below.

```r
finalTestPred <- predict(model, test)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(finalTestPred)

# finalTestPred ##prints out 20 problem_id_i.text files corresponding each class prediction.
```
