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


### C5.0 decision tree----
modelLookup("C5.0")

detectCores()
registerDoParallel(4)
getDoParWorkers()

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, selectionFunction = "best")

grid <- expand.grid(.model = c("tree","rules"),
                    .trials = seq(20, 70, by = 5),
                    .winnow = c("FALSE", "TRUE"))

system.time(m_C5.0 <- train(Survived ~ ., data=training, method="C5.0", metric="Accuracy", 
                            tuneGrid=grid, trControl=ctrl))
m_C5.0

survived_pred_training_prob_C5.0 <- predict(m_C5.0, training, type = 'prob')
survived_predictions_training <- predict(m_C5.0, training)
table(survived_predictions_training, training$Survived)
sum(diag(table(survived_predictions_training, training$Survived)))/sum(table(survived_predictions_training, training$Survived))
# best = 0.8967452

#plot(m_C5.0$finalModel) # virker ikke!
#text(m_C5.0$finalModel)

# final model:
grid_best <- expand.grid(.model = m_C5.0$bestTune[[2]],
                         .trials = m_C5.0$bestTune[[1]],
                         .winnow = m_C5.0$bestTune[[3]])  

best_model <- train(Survived ~ ., data=training, method='C5.0', metric="Accuracy", tuneGrid=grid_best)

#plot(best_model$finalModel)

survived_predictions_training_C5.0 <- predict(best_model, training)
table(survived_predictions_training_C5.0, training$Survived)
sum(diag(table(survived_predictions_training_C5.0, training$Survived)))/sum(table(survived_predictions_training_C5.0, training$Survived))

survived_predictions_test_C5.0 <- predict(best_model, test)
survived_predictions_test_C5.0_prob <- predict(best_model, test, type = 'prob')

# eksporterer predictions
library(rio)

C5.0_model <- as.data.frame(matrix(NA, ncol=2, nrow=nrow(test)))

C5.0_model[,2] <- as.character(survived_predictions_test_C5.0)
C5.0_model[,1] <-  as.character(testdata$PassengerId)
names(C5.0_model) <-  c('PassengerId', 'Survived')

export(C5.0_model, "~/Dropbox/R/Machine learning med R/Titanic/predictions/C5.0_model.csv")

registerDoParallel(1)



