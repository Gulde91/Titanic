### Indlæser og behandler data----

load('~/Dropbox/R/Machine learning med R/Titanic/træningsdata.RData')
load('~/Dropbox/R/Machine learning med R/Titanic/testdata.RData')
testdata$Survived <- NA

data <- rbind(træningsdata,testdata)

## data pre-processing 
data$Survived <- as.factor(data$Survived)

# indsætter for manglende for Embarked
data$Embarked[is.na(data$Embarked)] <- 'manglende'
data$Embarked <- as.factor(data$Embarked)

# Dummy ift om personen har en kabine
data$kabine <- as.factor(ifelse(is.na(data$Cabin), 0, 1))

# Finder titel
data$status <- substr(data$Name, as.numeric(regexec(',', data$Name))+2, nchar(data$Name))
data$status <- substr(data$status, 1, as.numeric(regexec(' ', data$status))-2)
data$status <- ifelse(data$status %in% c('Capt','Dr','Master','Miss','Mr','Mrs', 'Rev', 'Sir'), data$status, 'tom')


# data$age_group <- ifelse(is.na(data$Age), 'Ukendt', 
#                     ifelse(data$Age<5, 'baby',
#                       ifelse(data$Age<13, 'barn',
#                         ifelse(data$Age<18, 'teen',
#                           ifelse(data$Age<25, 'Ung',
#                             ifelse(data$Age<50, 'voksen',
#                               ifelse(data$Age<60, 'Ældre','gammel')))))))

# ekstrapolere Fare
fit <- lm(Fare~Pclass+Sex+SibSp+Parch+Embarked, data = data[!is.na(data$Fare),])

for (i in 1:nrow(data)) {
  if (is.na(data$Fare[i])) {
    tmp <- predict(fit, data[i,])
     if (tmp < min(data$Fare, na.rm = T)) {
       data$Fare[i] <- min(data$Fare, na.rm = T)
     }
     else if (tmp > max(data$Fare, na.rm = T)) {
       data$Fare[i] <- max(data$Fare, na.rm = T)
     }
    else {
      data$Fare[i] <-  tmp
    }
  }
}


# ekstrapolere Alder
fit <- lm(Age~Pclass+Sex+SibSp+Embarked+Embarked+status, data = data[!is.na(data$Age),])

for (i in 1:nrow(data)) {
  if (is.na(data$Age[i])) {
    tmp <- predict(fit, data[i,])
    if (tmp < min(data$Age, na.rm = T)) {
      data$Age[i] <- min(data$Age, na.rm = T)
    }
    else if (tmp > max(data$Age, na.rm = T)) {
      data$Age[i] <- max(data$Age, na.rm = T)
    }
    else {
      data$Age[i] <-  round(tmp)
    }
  }
}


# fjerne irrelevante variable
data1 <- subset(data, select = -c(PassengerId, Name, Ticket, Cabin)) # Age

## splitter data
summary(data1)

training <-  data1[1:891,]
test <-  data1[892:1309,]

rm(træningsdata, tmp, i, fit)

# skalerer data
tmp <-  rbind(training, test)
tmp$Sex <- as.numeric(factor(tmp$Sex, levels = c('male','female'), labels = c(2, 1)))
tmp$Embarked <- as.numeric(factor(tmp$Embarked, levels = c('S', 'C', 'Q', 'manglende'), labels = c(1,2,3,4)))
#tmp$age_group <- as.numeric(factor(tmp$age_group, levels = c('Ukendt','baby','barn','teen','Ung','voksen',
#                                                             'Ældre','gammel'), labels = c(1,2,3,4,5,6,7,8)))
tmp$kabine <- as.numeric(factor(tmp$kabine, levels = c('0','1'), 
                                labels = c(1,2)))
tmp$status <- as.numeric(factor(tmp$status, levels = c('Capt','Dr','Master','Miss','Mr','Mrs','Rev','Sir','tom'),
                                labels = c(1,2,3,4,5,6,7,8,9)))

tmp[-1] <- scale(tmp[-1]) # standardiserer 
training_scale <- tmp[1:891,]
test_scale <- tmp[892:1309,]

rm(tmp)


#### caret pakke ####
library(caret)
library(doParallel)
#

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



### Random Forrest----
library(randomForest)
modelLookup("RandomForest")

detectCores()
registerDoParallel(4)
getDoParWorkers()

#ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, selectionFunction = "best")
ctrl <- trainControl(method = "oob") # er hurtigere og giver valid accuarcy

grid <- expand.grid(.mtry=c(4,8)) # c(1:(ncol(training)-1))

system.time(m_rf <- train(Survived ~ ., data=training, method="rf", metric="Accuracy", 
                       tuneGrid=grid, trControl=ctrl, ntree = 500)) #, importance = T, verbose = T
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

### Neural netværk med h2o (virker pt ikke)----
install.packages('h2o')
library(h2o)

training_scale$Survived <- as.numeric(training_scale$Survived)

h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Survived',
                               training_frame = as.h2o(training_scale),
                               activation = 'Rectifier',
                               hidden = c(7,7),
                               epochs = 100,
                               train_samples_per_iteration = -2)

prop_pred = h2o.predict(classifier, as.h2o(training_scale[-1]))
survived_pred = (prop_pred>0.5)
survived_pred = as.vector(survived_pred)

table(survived_pred, training_scale$Survived)

table(survived_predictions_training_, training_scale$Survived)
sum(diag(table(survived_predictions_training_, training_scale$Survived)))/sum(table(survived_predictions_training_, training_scale$Survived))

survived_predictions_test_rf <- predict(best_model, test)



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














### test ----

prop_training <- as.data.frame(cbind(
  training$Survived,
  C5.0_prob = survived_pred_training_prob_C5.0[,2],
  nnet_prob = survived_pred_training_prob_nnet[,2],
  rf_prob = survived_pred_training_rf_prob[,2],
  xgboost_prob = survived_pred_training_xgb_prob[,2]
))

survived_pred_training_prob  
  
prop_test <- as.data.frame(cbind(
  C5.0_prob = survived_predictions_test_C5.0_prob[,2],
  nnet_prob = survived_predictions_test_nnet_prob[,2],
  rf_prob = survived_predictions_test_rf_prob[,2],
  xgboost_prob = survived_pred_test_xgboost_prob[,2]
  ))

test$survived <- NA

# prøver SVM
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, selectionFunction = "best")
grid <- expand.grid(.C = c(10^(-1:3)),
                    .sigma = c(2^(-6:2))) # 

system.time(m <- train(Survived ~ ., data = training_scale, method = 'svmRadial', metric = "Accuracy", trControl = ctrl, tuneGrid = grid))

m


library(rio)

test_model <- as.data.frame(matrix(NA, ncol=2, nrow=nrow(test)))

test_model[,2] <- as.character(test$survived)
test_model[,1] <-  as.character(testdata$PassengerId)
names(test_model) <-  c('PassengerId', 'Survived')

export(test_model, "~/Dropbox/R/Machine learning med R/Titanic/predictions/test_model.csv")




#### mere pre-processing (se http://topepo.github.io/caret/pre-processing.html) ----
library(caret)

# laver om til dummies
data_tmp <- data

dummies <- dummyVars(Survived ~ ., data = data_tmp)
dummies <- predict(dummies, newdata = data_tmp)

# tjekker om nogle variable har varians = 0 eller tæt på
nzv <- nearZeroVar(dummies, saveMetrics= TRUE)
nzv[nzv$nzv,]

# skalerer


data_pre <-  as.data.frame(cbind(Survived=data_tmp$Survived, dummies))
data_pre$Survived <- as.factor(data_pre$Survived)



training_pre <-  data_pre[1:891,]
test_pre <-  data_pre[892:1309,]


rm(dummies, data_tmp)


