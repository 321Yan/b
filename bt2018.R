library(randomForest)
library(xgboost)
library(mlr)
library(dplyr)
library(ggplot2)
source("D:/sfu/kaggle/b/b/fs.R")


rm(list=ls())
set.seed(123)
setwd("D:/sfu/kaggle/b")

samples = read.csv("b2018.csv",header = TRUE)
samples = subset(samples, select = - Record)

# extract test set
testIndex = sapply(samples$Sample,function(x){x=="Holdout"})
testData = samples[testIndex,!colnames(samples)%in%c("SUBSCRIBE","Sample")]
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
  
  if("SUBSCRIBE"%in%colnames(test)) {
    test_y = test[,"SUBSCRIBE"]
    test = subset(test, select = -SUBSCRIBE)
  }
  if("SUBSCRIBE"%in%colnames(train)) {
    train_y = train[,"SUBSCRIBE"]
    train = subset(train, select = -SUBSCRIBE)
  }
  
  n_train = nrow(train)
  n_test = nrow(test)
  n_tot = n_train+n_test
  
  tot = rbind(train,test)
  
  # be aware of the na's (created by 1900 or as.Date)
  tot$LastOrder = as.numeric(max(as.Date(tot$LastOrder, format = "%m/%d/%Y"),na.rm = T)-as.Date(tot$LastOrder, format = "%m/%d/%Y"))
  DiscImp = sapply(tot$Disc, is.na)
  TitleImp = sapply(tot$Title, is.na)
  Weeks3Meals = sapply(tot$Weeks3Meals,is.na)
  
  # tot[,sapply(tot,is.factor)] = as.numeric(tot[,sapply(tot,is.factor)])
  

  tot = createDummyFeatures(tot)
  # tot[,15:22] = as.data.frame(lapply(tot[,15:22],as.integer))
  prepro = caret::preProcess(tot,method = "bagImpute")
  tot = predict(prepro,tot)
  tot$DiscImp = DiscImp
  tot$TitleImp = TitleImp
  tot$Weeks3Meals= Weeks3Meals
  
  
  
  # tot = randomForest::na.roughfix(tot)
  
  train = cbind(tot[1:n_train,],"SUBSCRIBE" = train_y)
  
  if(option == 0){
    test = cbind(tot[(n_train+1):n_tot, ], "SUBSCRIBE" = test_y)
    tot = rbind(train,test)
    return(tot)
    
  } else if(option == 1) {
    
    test = tot[(n_train+1):n_tot, ]
    
    return(list(train,test))
    
  }
  
}


tot = data_prep(train = train, test = test,option = 0)


