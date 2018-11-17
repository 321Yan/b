library(randomForest)
library(xgboost)
library(mlr)
library(dplyr)
library(ggplot2)
source("D:/sfu/kaggle/b/b/fs.R")


rm(list=ls())
set.seed(123)
setwd("D:/sfu/kaggle/b")

samples = read.csv("b.csv",header = TRUE)

# extract test set
testIndex = sapply(samples$Sample,function(x){x=="Holdout"})
testData = samples[testIndex,!colnames(samples)%in%c("lost","Sample")]
testId = testData$custid
testData = subset(testData, select = -c(custid,esent))
# testData = subset(testData, select = -custid)

# no need for estimation/validation label
samples = samples[!testIndex,!colnames(samples)%in%"Sample"]
samplesId = samples$custid
samples = subset(samples, select = -c(custid,esent))
# samples = subset(samples, select = -custid)


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

# p = ggplot(tot, aes(x=esent, y=lost)) +
#   ggtitle("scatter plot")
# p1 = p + geom_point(alpha = 0.01, colour = "orange") +
#   geom_density2d() + theme_bw()
# plot(p1)




tsk = makeClassifTask(data = tot, target = "lost")
# View(listLearners(tsk, properties = "prob"))

# split data into train and test
h = makeResampleDesc("Holdout")
ho = makeResampleInstance(h,tsk)
tsk.train = subsetTask(tsk,ho$train.inds[[1]])
tsk.test = subsetTask(tsk,ho$test.inds[[1]])

# use all cpus during training
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores()-1)

# number of iterations used for hyperparameters tuning
tc = makeTuneControlRandom(maxit = 20)

# resampling strategy for evaluating model performance
# rdesc = makeResampleDesc("RepCV", reps = 2, folds = 3)
rdesc = makeResampleDesc("CV", iters = 3)





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
r = resample(rf_lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = list(tpr,fpr,fnr,tnr,f1,acc))
rf_mod = train(rf_lrn, tsk.train)
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
performance(gbm_pred, measures = acc)
r = resample(gbm_lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = list(tpr,fpr,fnr,tnr,f1,acc))
plotFeatureImportance(gbm_mod,10)
#-------------------------------------

#------------------ svm_radial ------------------
svm_lrn = makeLearner(cl = "classif.svm", predict.type = "prob", par.vals = list())
svm_ps = makeParamSet( makeNumericParam("gamma",lower = 0.001, upper= 0.1),makeNumericParam("cost",lower = 1,upper = 10),
                       makeNumericParam("tolerance",lower = 0.0005 ,upper = 0.01))
svm_tr = tuneParams(svm_lrn,tsk.train,cv3,acc,svm_ps,tc)
svm_lrn = setHyperPars(svm_lrn,par.vals = svm_tr$x)

svm_mod = train(svm_lrn, tsk.train)
svm_pred = predict(svm_mod, tsk.test)
performance(svm_pred, measures = acc)
r = resample(svm_lrn, tsk, resampling = rdesc, show.info = T, models = F,measures =list(tpr,fpr,fnr,tnr,f1,acc))

#-------------------------------------
#------------------ xgboost_tree ------------------
xgb_train = tot[1:nrow(train),]
xgb_test = tot[(nrow(train)+1):nrow(samples),]

#using one hot encoding 
xgb_train_y <- xgb_train$lost 
xgb_test_y <- xgb_test$lost
xgb_train <- as.matrix(subset(xgb_train,select=-lost))
xgb_test <- as.matrix(subset(xgb_test,select=-lost))

#convert factor to numeric 
xgb_train_y<- as.numeric(xgb_train_y)-1
xgb_test_y  <- as.numeric(xgb_test_y )-1

dtrain = xgb.DMatrix(data = xgb_train,label = xgb_train_y) 
dtest = xgb.DMatrix(data =xgb_test,label= xgb_test_y )

params <- list(booster = "gbtree",
               objective = "binary:logistic", eta=0.08, gamma=0, max_depth=5, min_child_weight=1, subsample=0.8, colsample_bytree=0.8,nthread = 8)
xgbcv <- xgb.cv( params = params, data =dtrain, nrounds = 80, nfold = 5, showsd = T, stratified = T,
                 print_every_n = 10, early_stop_round = 20, maximize = F,metrics = 'error')

# tuning
xgb_lrn = makeLearner(cl = "classif.xgboost",predict.type = "prob")
xgb_lrn$par.vals = list(objective="binary:logistic", eval_metric="error", nrounds=80, eta=0.08,stratified = T,verbose=0)
xgb_ps = makeParamSet( makeIntegerParam("max_depth",lower = 7,upper = 14),
                       makeNumericParam("min_child_weight",lower = 1,upper = 9), makeNumericParam("subsample",lower = 0.5,upper = 1),
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
xgb_tr = tuneParams(xgb_lrn,tsk.train,cv3,acc,xgb_ps,tc)
xgb_lrn = setHyperPars(xgb_lrn,par.vals = xgb_tr$x)

xgb_mod = train(xgb_lrn, tsk.train)
xgb_pred = predict(xgb_mod, tsk.test)
performance(xgb_pred, measures = acc)
r = resample(xgb_lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = list(tpr,fpr,fnr,tnr,f1,acc))

#------------------------------------------------

#------------------ xgboost_linear ------------------
# xgb_train = tot[1:nrow(train),]
# xgb_test = tot[(nrow(train)+1):nrow(samples),]
# 
# #using one hot encoding 
# xgb_train_y <- xgb_train$lost 
# xgb_test_y <- xgb_test$lost
# xgb_train <- as.matrix(subset(xgb_train,select=-lost))
# xgb_test <- as.matrix(subset(xgb_test,select=-lost))
# 
# #convert factor to numeric 
# xgb_train_y<- as.numeric(xgb_train_y)-1
# xgb_test_y  <- as.numeric(xgb_test_y )-1
# 
# dtrain = xgb.DMatrix(data = xgb_train,label = xgb_train_y) 
# dtest = xgb.DMatrix(data =xgb_test,label= xgb_test_y )
# 
# params <- list(booster = "gblinear",
#                objective = "binary:logistic", eta=0.1, gamma=0, max_depth=5, min_child_weight=1, subsample=0.8, colsample_bytree=0.8,nthread = 8)
# xgbcv <- xgb.cv( params = params, data =dtrain, nrounds = 61, nfold = 5, showsd = T, stratified = T,
#                  print_every_n = 10, early_stop_round = 20, maximize = F,metrics = 'error')
# 
# # tuning
# xgbl_lrn = makeLearner(cl = "classif.xgboost",predict.type = "prob")
# xgbl_lrn$par.vals = list(booster = "gblinear", objective="binary:logistic", eval_metric="error", nrounds=61, eta=0.1,stratified = T,verbose=0)
# # xgbl_lrn = makePreprocWrapperCaret(xgbl_lrn, ppc.pca = TRUE, ppc.thresh = 1)
# xgbl_ps = makeParamSet( makeNumericParam("lambda",lower = 0,upper = 1), makeNumericParam("subsample",lower = 0.5,upper = 1),
#                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
# xgbl_tr = tuneParams(xgbl_lrn,tsk.train,cv3,acc,xgb_ps,tc)
# xgbl_lrn = setHyperPars(xgbl_lrn,par.vals = xgbl_tr$x)
# 
# xgbl_mod = train(xgbl_lrn, tsk.train)
# xgbl_pred = predict(xgbl_mod, tsk.test)
# performance(xgbl_pred, measures = acc)
# r = resample(xgbl_lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = list(tpr,fpr,fnr,tnr,f1,acc))
#------------------------------------------------


#------------------ ensemble --------------------

m = makeStackedLearner(base.learners = list(rf_lrn,xgb_lrn,gbm_lrn),
                       predict.type = "prob", method = 'hill.climb')

#------------------------------------------------

#------------------ submimssion -----------------


sub = data_prep(train=samples,test = testData,option = 1)
sub = sub[[2]]

make_prediction = function(lrn,tsk,sub_data,subname) {
  mod = train(lrn,tsk)
  pred = predict(mod,newdata = sub_data)
  
  
  rdesc = makeResampleDesc("RepCV", reps = 3, folds = 3)
  r = resample(lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = acc)
  
  submission = data.frame(custid = testId, Score = NA)
  submission$Score = pred$data$response
  write.csv(submission,file = subname,row.names = F, col.names = T)
  
}

make_prediction(lrn = rf_lrn,tsk = tsk,sub_data = sub,subname = "rf_-planet.csv")
make_prediction(lrn = xgb_lrn,tsk = tsk,sub_data = sub,subname = "xgb_-planet.csv")
make_prediction(lrn = gbm_lrn,tsk = tsk,sub_data = sub,subname = "gbm_-planet.csv")
make_prediction(m,tsk = tsk,sub_data = sub,subname = "ens_-planet.csv")
