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

### Random Forest----
library(randomForest)
modelLookup("rf")

detectCores()
registerDoParallel(4)
getDoParWorkers()

#ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, selectionFunction = "best")
ctrl <- trainControl(method = "oob") # er hurtigere og giver valid accuarcy

grid <- expand.grid(.mtry=c(1:(ncol(training)-1))) # c(1:(ncol(training)-1))

system.time(m_rf <- train(Survived ~ ., data=training, method="rf", metric="Accuracy", 
                          tuneGrid=grid, trControl=ctrl, ntree = 500, importance = T, verbose = T)) #
m_rf

varImp(m_rf, scale = F)

# Se randomForest approach fra ISLR bog for bedre variable importance metode  

survived_pred_training_rf_prob <- predict(m_rf, training, type='prob')
survived_predictions_training <- predict(m_rf, training)
table(survived_predictions_training, training$Survived)
sum(diag(table(survived_predictions_training, training$Survived)))/sum(table(survived_predictions_training, training$Survived))

# final model:
grid_best <- expand.grid(.mtry = m_rf$bestTune[[1]])  

system.time(best_model <- train(Survived ~ ., data=training, method="rf", metric="Accuracy", 
                                tuneGrid=grid_best, ntree = 500))

survived_predictions_training_rf <- predict(best_model, training)
table(survived_predictions_training_rf, training$Survived)
sum(diag(table(survived_predictions_training_rf, training$Survived)))/sum(table(survived_predictions_training_rf, training$Survived))

survived_predictions_test_rf <- predict(best_model, test)
survived_predictions_test_rf_prob <- predict(best_model, test, type='prob')

# eksporterer predictions
library(rio)

rf_model <- as.data.frame(matrix(NA, ncol=2, nrow=nrow(test)))

rf_model[,2] <- as.character(survived_predictions_test_rf)
rf_model[,1] <-  as.character(testdata$PassengerId)
names(rf_model) <-  c('PassengerId', 'Survived')

export(rf_model, "~/Dropbox/R/Machine learning med R/Titanic/predictions/rf_model.csv")

registerDoParallel(1)

