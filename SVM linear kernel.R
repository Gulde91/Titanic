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


### Support Vector Machines with Linear Kernel ----
library(kernlab)

detectCores()
registerDoParallel(4)
getDoParWorkers()

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats=2, selectionFunction = "best")
grid <- expand.grid(.C = c(10^(-2:3)))

system.time(m <- train(Survived ~ ., data = training_scale, method = 'svmLinear', metric = "Accuracy", trControl = ctrl, tuneGrid = grid))

m

survived_predictions_training <- predict(m, training_scale)
table(survived_predictions_training, training_scale$Survived)
sum(diag(table(survived_predictions_training, training_scale$Survived)))/sum(table(survived_predictions_training, training_scale$Survived))

# final model:
grid_best <- expand.grid(.C = m$bestTune[[1]])

best_model <- train(Survived ~ ., data=training_scale, method='svmLinear', metric="Accuracy", tuneGrid=grid_best)

survived_predictions_training_svm_linear <- predict(best_model, training_scale)
table(survived_predictions_training_svm_linear, training_scale$Survived)
sum(diag(table(survived_predictions_training_svm_linear, training_scale$Survived)))/sum(table(survived_predictions_training_svm_linear, training_scale$Survived))

survived_predictions_test_svm_linear <- predict(m, test_scale)

# eksporterer predictions
install.packages('rio')
library(rio)

svm_linear_model <- as.data.frame(matrix(NA, ncol=2, nrow=nrow(test_scale)))

svm_linear_model[,2] <- as.character(survived_predictions_test_svm_linear)
svm_linear_model[,1] <-  as.character(testdata$PassengerId)
names(svm_linear_model) <-  c('PassengerId', 'Survived')

export(svm_linear_model, "~/Dropbox/R/Machine learning med R/Titanic/predictions/svm_linear_model.csv")

