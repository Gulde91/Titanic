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



### XGBoost ----
library(xgboost)
library(plyr)
modelLookup("xgbTree")

detectCores()
registerDoParallel(4)
getDoParWorkers()

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, selectionFunction = "best")

grid <- expand.grid(.nrounds = c(50, 100, 150, 200),
                    .max_depth = c(1:5),
                    .eta = c(0.2, 0.3, 0.4, 0.5),
                    .gamma = c(0, 0.5, 1),
                    .colsample_bytree = c(0.5, 0.6, 0.7, 0.8, 0.9, 1),
                    .min_child_weight = 1,
                    .subsample = c(0.5, 0.75, 1))

system.time(m_xgb <- train(Survived ~ ., data=training, method="xgbTree", metric="Accuracy", 
                           trControl=ctrl)) # tuneGrid=grid, 
m_xgb

survived_pred_training_xgb_prob<- predict(m_xgb, training, type = 'prob')
survived_predictions_training <- predict(m_xgb, training)
table(survived_predictions_training, training$Survived)
sum(diag(table(survived_predictions_training, training$Survived)))/sum(table(survived_predictions_training, training$Survived))

# final model:
grid_best <- expand.grid(.nrounds = m_xgb$bestTune[[1]],
                         .max_depth = m_xgb$bestTune[[2]],
                         .eta = m_xgb$bestTune[[3]],
                         .gamma = m_xgb$bestTune[[4]],
                         .colsample_bytree = m_xgb$bestTune[[5]],
                         .min_child_weight = m_xgb$bestTune[[6]],
                         .subsample = m_xgb$bestTune[[7]])

system.time(best_model <- train(Survived ~ ., data=training, method="xgbTree", metric="Accuracy", 
                                tuneGrid=grid_best))

survived_predictions_training_xgboost <- predict(best_model, training)
table(survived_predictions_training_xgboost, training$Survived)
sum(diag(table(survived_predictions_training_xgboost, training$Survived)))/sum(table(survived_predictions_training_xgboost, training$Survived))

survived_predictions_test_xgboost <- predict(best_model, test)
survived_pred_test_xgboost_prob <- predict(best_model, test, type='prob')

# eksporterer predictions
library(rio)

xgboost_model <- as.data.frame(matrix(NA, ncol=2, nrow=nrow(test)))

xgboost_model[,2] <- as.character(survived_predictions_test_xgboost)
xgboost_model[,1] <-  as.character(testdata$PassengerId)
names(xgboost_model) <-  c('PassengerId', 'Survived')

export(xgboost_model, "~/Dropbox/R/Machine learning med R/Titanic/predictions/xgboost_model.csv")

registerDoParallel(1)







