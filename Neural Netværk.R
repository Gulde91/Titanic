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




### Neural Netværk ----
library(caret)
library(doParallel)
modelLookup('nnet')

detectCores()
registerDoParallel(4)
getDoParWorkers()

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, selectionFunction = "best")

grid <- expand.grid(.decay = c(seq(0.1, 0.5, by=0.05)), 
                    .size = c(1:10))

system.time(m_nnet <- train(Survived ~ ., data=training_scale, method="nnet", metric="Accuracy", 
                            tuneGrid=grid, trControl=ctrl, maxit=1000))
m_nnet

survived_pred_training_prob_nnet <- predict(m_nnet, training_scale, type = 'prob')
survived_predictions_training <- predict(m_nnet, training_scale)
table(survived_predictions_training, training_scale$Survived)
sum(diag(table(survived_predictions_training, training_scale$Survived)))/sum(table(survived_predictions_training, training_scale$Survived))

# final model:
grid_best <- expand.grid(.decay = m_nnet$bestTune[[2]],
                         .size = m_nnet$bestTune[[1]]) 

best_model <- train(Survived ~ ., data=training_scale, method='nnet', metric="Accuracy", 
                    tuneGrid=grid_best, maxit=1000) 

#plot(best_model$finalModel) # virker ikke, prøv: https://beckmw.wordpress.com/2013/03/04/visualizing-neural-networks-from-the-nnet-package/

survived_predictions_training_nnet <- predict(best_model, training_scale)
table(survived_predictions_training_nnet, training_scale$Survived)
sum(diag(table(survived_predictions_training_nnet, training_scale$Survived)))/sum(table(survived_predictions_training_nnet, training_scale$Survived))

survived_predictions_test_nnet <- predict(best_model, test_scale)
survived_predictions_test_nnet_prob <- predict(best_model, test_scale, type='prob')

# eksporterer predictions
library(rio)

nnet_model <- as.data.frame(matrix(NA, ncol=2, nrow=nrow(test)))

nnet_model[,2] <- as.character(survived_predictions_test_nnet)
nnet_model[,1] <-  as.character(testdata$PassengerId)
names(nnet_model) <-  c('PassengerId', 'Survived')

export(nnet_model, "~/Dropbox/R/Machine learning med R/Titanic/predictions/nnet_model.csv")

registerDoParallel(1)



