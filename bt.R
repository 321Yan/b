library(randomForest)
library(xgboost)
library(mlr)
library(dplyr)



rm(list=ls())
set.seed(123)
setwd("D:/sfu/kaggle/b")

samples = read.csv("b.csv",header = TRUE)

# extract test set
testIndex = sapply(samples$Sample,function(x){x=="Holdout"})
testData = samples[testIndex,!colnames(samples)%in%c("lost","Sample")]
testId = testData$custid
testData = subset(testData, select = -custid)

# no need for estimation/validation label
samples = samples[!testIndex,!colnames(samples)%in%"Sample"]
samplesId = samples$custid
samples = subset(samples, select = -custid)



# report missing data
missing = as.vector(NULL)
for(i in 1:ncol(samples)) {
  n = sum(is.na(samples[,i]))
  p = round(n/nrow(samples),2)
  if(n > 0) {
    print(paste0(colnames(samples[i]),": ",n," (",p,")"))
    missing = c(missing,colnames(samples[i]))
  }
}


index = sample(nrow(samples),floor(0.75*nrow(samples)))
train = samples[index,]
test = samples[-index,]

data_prep = function(train, test, option) {
  
  if("lost"%in%colnames(test)) {
    test_y = test[,"lost"]
    test = subset(test, select = -lost)
  }
  if("lost"%in%colnames(train)) {
    train_y = train[,"lost"]
    train = subset(train, select = -lost)
  }
  
  n_train = nrow(train)
  n_test = nrow(test)
  n_tot = n_train+n_test
  
  tot = rbind(train,test)

  # be aware of the na's (created by 1900 or as.Date)
  tot$created = as.numeric(max(as.Date(tot$created, format = "%d/%m/%Y"),na.rm = T)-as.Date(tot$created, format = "%d/%m/%Y"))
  tot$firstorder = as.numeric(max(as.Date(tot$firstorder, format = "%d/%m/%Y"),na.rm = T)-as.Date(tot$firstorder, format = "%d/%m/%Y"))
  tot$lastorder = as.numeric(max(as.Date(tot$lastorder, format = "%d/%m/%Y"),na.rm = T)-as.Date(tot$lastorder, format = "%d/%m/%Y"))
  
  
  # tot[,sapply(tot,is.factor)] = as.numeric(tot[,sapply(tot,is.factor)])
  
  ### remove outliers
  # ...
  tot = createDummyFeatures(tot)
  prepro = caret::preProcess(tot,method = "bagImpute")
  tot = predict(prepro,tot)
  
  # tot = randomForest::na.roughfix(tot)
  
  train = cbind(tot[1:n_train,],"lost" = train_y)
  
  if(option == 0){
    test = cbind(tot[(n_train+1):n_tot, ], "lost" = test_y)
    tot = rbind(train,test)
    return(tot)
    
  } else if(option == 1) {
    
    test = tot[(n_train+1):n_tot, ]
    
    return(list(train,test))
    
  }
  
}


tot = data_prep(train = train, test = test,option = 0)



