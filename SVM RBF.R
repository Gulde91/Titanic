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


### Support Vector Machines with Radial Basis Function Kernel----
library(kernlab)
library(doParallel)

detectCores()
registerDoParallel(4)
getDoParWorkers()

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, selectionFunction = "best")
grid <- expand.grid(.C = c(10^(-1:3)),
                    .sigma = c(2^(-6:2))) # 

system.time(m_svm <- train(Survived ~ ., data = training_scale, method = 'svmRadial', metric = "Accuracy", trControl = ctrl, tuneGrid = grid))

m_svm

survived_predictions_training <- predict(m_svm, training_scale)
table(survived_predictions_training, training_scale$Survived)
sum(diag(table(survived_predictions_training, training_scale$Survived)))/sum(table(survived_predictions_training, training_scale$Survived))
# best = 0.8619529

# final model: 
grid_best <- expand.grid(.C = m_svm$bestTune[[2]],
                         .sigma = m_svm$bestTune[[1]]) # 

best_model <- train(Survived ~ ., data=training_scale, method='svmRadial', metric="Accuracy", tuneGrid=grid_best)

survived_predictions_training_svm_rbf <- predict(best_model, training_scale)
table(survived_predictions_training_svm_rbf, training_scale$Survived)
sum(diag(table(survived_predictions_training_svm_rbf, training_scale$Survived)))/sum(table(survived_predictions_training_svm_rbf, training_scale$Survived))

survived_predictions_test_svm_rbf <- predict(best_model, test_scale)

# eksporterer predictions 
library(rio)

svm_RBF_model <- as.data.frame(matrix(NA, ncol=2, nrow=nrow(test_scale)))

svm_RBF_model[,2] <- as.character(survived_predictions_test_svm_rbf)
svm_RBF_model[,1] <-  as.character(testdata$PassengerId)
names(svm_RBF_model) <-  c('PassengerId', 'Survived')

export(svm_RBF_model, "~/Dropbox/R/Machine learning med R/Titanic/predictions/svm_RBF_model.csv")


