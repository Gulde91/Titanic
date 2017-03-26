### henter data ----
rm(list = ls())

load('~/Dropbox/R/Machine learning med R/Titanic/træningsdata.RData')
load('~/Dropbox/R/Machine learning med R/Titanic/testdata.RData')

source('~/Dropbox/R/Machine learning med R/Titanic/data_prep.R')

data <- getdata(træningsdata=træningsdata, testdata=testdata)
training <- data[[1]]
test <- data[[2]]
training_scale <- data[[3]]
test_scale <- data[[4]]

rm(data, getdata)

### caret pakke ----
library(caret)
library(doParallel)



### Naive Baysian----
library(e1071)

data_NB_training <- as.data.frame(cbind(Survived = as.factor(training$Survived), #svm_linear = survived_predictions_training_svm_linear,
                                        svm_rbf = survived_predictions_training_svm_rbf, C5.0 = survived_predictions_training_C5.0, 
                                        nnet = survived_predictions_training_nnet, rf = survived_predictions_training_rf,
                                        xgboost = survived_predictions_training_xgboost))

data_NB_training$Survived <- as.factor(data_NB_training$Survived)

data_NB_test <- as.data.frame(cbind(#svm_linear = survived_predictions_test_svm_linear,
  svm_rbf = survived_predictions_test_svm_rbf,
  C5.0 = survived_predictions_test_C5.0,
  nnet = survived_predictions_test_nnet,
  rf = survived_predictions_test_rf,
  xgboost = survived_predictions_test_xgboost))

m <- naiveBayes(Survived~., data = data_NB_training, laplace = 1)

survived_predictions_test_NB <- predict(m, data_NB_training)
table(survived_predictions_test_NB, data_NB_training$Survived)
sum(diag(table(survived_predictions_test_NB, data_NB_training$Survived)))/sum(table(survived_predictions_test_NB, data_NB_training$Survived))

survived_predictions_test_rf <- predict(m, data_NB_test)

# eksporterer predictions
library(rio)

NB_model <- as.data.frame(matrix(NA, ncol=2, nrow=nrow(test)))

NB_model[,2] <- as.character(survived_predictions_test_rf)
NB_model[,1] <-  as.character(testdata$PassengerId)
names(NB_model) <-  c('PassengerId', 'Survived')

NB_model$Survived[NB_model$Survived=='1'] <- 0
NB_model$Survived[NB_model$Survived=='2'] <- 1

export(NB_model, "~/Dropbox/R/Machine learning med R/Titanic/predictions/NB_model.csv")














