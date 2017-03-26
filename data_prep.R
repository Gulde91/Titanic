### Indlæser og behandler data----

getdata <- function(træningsdata, testdata) { 
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

list1 <- list(training, test, training_scale, test_scale)
return(list1)
}
