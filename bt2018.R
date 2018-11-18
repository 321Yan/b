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
samples = subset(samples, select = - Weeks3Meals)

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
  DiscImp = sapply(tot$Disc, function(x){as.numeric(is.na(x))})
  TitleImp = sapply(tot$Title, function(x){as.numeric(is.na(x))})

  
  # tot[,sapply(tot,is.factor)] = as.numeric(tot[,sapply(tot,is.factor)])
  

  tot = createDummyFeatures(tot)
  # tot[,15:22] = as.data.frame(lapply(tot[,15:22],as.integer))
  prepro = caret::preProcess(tot,method = "bagImpute")
  # prepro = caret::preProcess(tot,method = "knnImpute")
  tot = predict(prepro,tot)
  tot$DiscImp = DiscImp
  tot$TitleImp = TitleImp
  # tot$DA_age = tot$DA_Under20/(tot$DA_Under20+tot$DA_Over60)
  
  
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
# write.csv(tot, file = "Imputed.csv",row.names = F, col.names = T)

tsk = makeClassifTask(data = tot, target = "SUBSCRIBE")
# View(listLearners(tsk, properties = "prob"))

# split data into train and test
h = makeResampleDesc("Holdout")
ho = makeResampleInstance(h,tsk)
tsk.train = subsetTask(tsk,ho$train.inds[[1]])
tsk.test = subsetTask(tsk,ho$test.inds[[1]])

# use all cpus during training
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())

# number of iterations used for hyperparameters tuning
tc = makeTuneControlRandom(maxit = 50)

# resampling strategy for evaluating model performance
rdesc = makeResampleDesc("RepCV", reps = 2, folds = 3)
# rdesc = makeResampleDesc("CV", iters = 3)


#------------- user-defined metric used during resampling ----------------
f = function(task, model, pred,feats, extra.args){
  predictData = pred$data
  predictData = predictData[order(predictData$prob.Y,decreasing = T),]
  totalPositive = sum(predictData$truth=="Y")
  predictData = predictData[1:floor(0.4*nrow(predictData)),'truth']
  result = (1-sum(predictData == "Y")/totalPositive)
  return(result)
}

ccl = makeMeasure(id = "ccl", minimize = TRUE, properties = c("classif", "prob"), fun = f)
#------------------------------------------------------------------------


#------------------ randomForest ------------------
# build model
rf_lrn = makeLearner(cl ="classif.randomForest", predict.type = "prob",par.vals = list())
# define the search range of hyperparameters
rf_ps = makeParamSet( makeIntegerParam("ntree",150,600),makeIntegerParam("nodesize",lower = 3,upper = 15),
                      makeIntegerParam("mtry",lower = 2,upper = 20),makeLogicalParam("importance",default = FALSE))

# search for the best hyperparameters
rf_tr = tuneParams(rf_lrn,tsk.train,cv3,acc,rf_ps,tc)
# specify the hyperparmeters for the model
rf_lrn = setHyperPars(rf_lrn,par.vals = rf_tr$x)

# evaluate performance use CV
r = resample(rf_lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = list(tpr,fpr,fnr,tnr,f1,acc,ccl))

plotFeatureImportance(rf_mod)


#-------------------------------------

#------------------ gbm ------------------
gbm_lrn = makeLearner(cl = "classif.gbm", predict.type = "prob",par.vals = list())
gbm_ps = makeParamSet( makeNumericParam("shrinkage",lower = 0.0001, upper= 0.01),makeNumericParam("bag.fraction",lower = 0.5,upper = 1),
                       makeIntegerParam("n.trees",lower = 50,upper = 500), makeIntegerParam("interaction.depth",lower = 1,upper = 10),
                       makeIntegerParam("n.minobsinnode",lower = 5,upper = 30))
gbm_tr = tuneParams(gbm_lrn,tsk.train,cv3,acc,gbm_ps,tc)
gbm_lrn = setHyperPars(gbm_lrn,par.vals = gbm_tr$x)

gbm_mod = train(gbm_lrn, tsk.train)
gbm_pred = predict(gbm_mod, tsk.test)
r = resample(gbm_lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = list(tpr,fpr,fnr,tnr,f1,acc,ccl))


# plotFeatureImportance(gbm_mod)
# cumulativeCapturedLift(gbm_pred$data,"Y",0.4)
#-------------------------------------

#------------------ svm_radial ------------------
svm_lrn = makeLearner(cl = "classif.svm", predict.type = "prob", par.vals = list())
svm_ps = makeParamSet( makeNumericParam("gamma",lower = 0.001, upper= 0.1),makeNumericParam("cost",lower = 1,upper = 10),
                       makeNumericParam("tolerance",lower = 0.0005 ,upper = 0.01))
svm_tr = tuneParams(svm_lrn,tsk.train,cv3,acc,svm_ps,tc)
svm_lrn = setHyperPars(svm_lrn,par.vals = svm_tr$x)

svm_mod = train(svm_lrn, tsk.train)
svm_pred = predict(svm_mod, tsk.test)
r = resample(svm_lrn, tsk, resampling = rdesc, show.info = T, models = F,measures =list(tpr,fpr,fnr,tnr,f1,acc,ccl))
# cumulativeCapturedLift(svm_pred$data,"Y",0.4)
#-------------------------------------
#------------------ xgboost_tree ------------------
xgb_train = tot[1:nrow(train),]
xgb_test = tot[(nrow(train)+1):nrow(samples),]

#using one hot encoding 
xgb_train_y <- xgb_train$SUBSCRIBE 
xgb_test_y <- xgb_test$SUBSCRIBE
xgb_train <- as.matrix(subset(xgb_train,select=-SUBSCRIBE))
xgb_test <- as.matrix(subset(xgb_test,select=-SUBSCRIBE))

#convert factor to numeric 
xgb_train_y<- as.numeric(xgb_train_y)-1
xgb_test_y  <- as.numeric(xgb_test_y )-1

dtrain = xgb.DMatrix(data = xgb_train,label = xgb_train_y) 
dtest = xgb.DMatrix(data =xgb_test,label= xgb_test_y )

params <- list(booster = "gbtree",
               objective = "binary:logistic", eta=0.05, gamma=0, max_depth=5, min_child_weight=1, subsample=0.8, colsample_bytree=0.8,nthread = 8)
xgbcv <- xgb.cv( params = params, data =dtrain, nrounds = 70, nfold = 5, showsd = T, 
                 print_every_n = 10, early_stop_round = 20, maximize = F,metrics = 'error')

# tuning
xgb_lrn = makeLearner(cl = "classif.xgboost",predict.type = "prob")
xgb_lrn$par.vals = list(objective="binary:logistic", eval_metric="error", nrounds=70, eta=0.05, verbose=0)
xgb_ps = makeParamSet( makeIntegerParam("max_depth",lower = 7,upper = 14),
                       makeNumericParam("min_child_weight",lower = 1,upper = 9), makeNumericParam("subsample",lower = 0.5,upper = 1),
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
xgb_tr = tuneParams(xgb_lrn,tsk.train,cv3,acc,xgb_ps,tc)
xgb_lrn = setHyperPars(xgb_lrn,par.vals = xgb_tr$x)

xgb_mod = train(xgb_lrn, tsk.train)
xgb_pred = predict(xgb_mod, tsk.test)
r = resample(xgb_lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = list(tpr,fpr,fnr,tnr,f1,acc,ccl))
# plotFeatureImportance(xgb_mod)
# cumulativeCapturedLift(xgb_pred$data,"Y",0.4)


#------------------------------------------------



#------------------ ensemble --------------------
m = makeStackedLearner(base.learners = list(xgb_lrn,gbm_lrn,svm_lrn),
                       predict.type = "prob", method = "hill.climb")

#------------------------------------------------

#------------------ submimssion -----------------


sub = data_prep(train=samples,test = testData,option = 1)
sub = sub[[2]]
# write.csv(sub, file = "holdout.csv",row.names = F, col.names = T)
make_prediction = function(lrn,tsk,sub_data,subname) {
  mod = train(lrn,tsk)
  pred = predict(mod,newdata = sub_data)
  
  mod_t = train(lrn,tsk.train)
  pred_t = predict(mod_t, tsk.test)
  
  rdesc = makeResampleDesc("RepCV", reps = 3, folds = 3)
  r = resample(lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures =list(tpr,fpr,fnr,tnr,f1,acc))
  
  submission = data.frame(custid = testId, Score = NA)
  submission$Score = pred$data$prob.Y
  write.csv(submission,file = subname,row.names = F, col.names = T)
  return(pred_t)
  
}

# rfm = make_prediction(lrn = rf_lrn,tsk = tsk,sub_data = sub,subname = "rf.csv")
# cumulativeCapturedLift(rfm$data,"Y",0.4)

xgbm = make_prediction(lrn = xgb_lrn,tsk = tsk,sub_data = sub,subname = "xgb.csv")
cumulativeCapturedLift(xgbm$data,"Y",0.4)

gbmm = make_prediction(lrn = gbm_lrn,tsk = tsk,sub_data = sub,subname = "gbm.csv")
cumulativeCapturedLift(gbmm$data,"Y",0.4)

svmm = make_prediction(lrn = svm_lrn,tsk = tsk,sub_data = sub,subname = "svm.csv")
cumulativeCapturedLift(svmm$data,"Y",0.4)

ensm = make_prediction(m,tsk = tsk,sub_data = sub,subname = "ens.csv")
cumulativeCapturedLift(ensm$data,"Y",0.4)

