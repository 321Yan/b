


# get the feature importance plot of trained models from mlr package 
# only work for tree-based model
#------------------------ example ------------------------
# r = resample(rf_lrn, tsk, resampling = rdesc, show.info = T, models = T,measures = mae)
# plotFeatureImportance(r$models[[1]],10)

# --- or ---
# rf_mod = train(rf_lrn, tsk.train)
# plotFeatureImportance(rf_mod)
#---------------------------------------------------------
plotFeatureImportance = function(model,topn = NULL) {
  f = getFeatureImportance(model)
  imp = f$res
  d = data.frame(Features = names(imp), Importance = as.numeric(imp))
  d = arrange(d,desc(Importance))
  if (is.null(topn)) {
    p = ggplot(d,aes(Features,Importance))+geom_col()+
      scale_x_discrete(limits=d$Features)+
      labs(title = paste0("Feature Importance of ",model$learner$name," Model"))+
      theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(angle = 45, hjust = 1,size = 12))
  } else {
    p = ggplot(d[1:topn,],aes(Features,Importance))+geom_col()+
      scale_x_discrete(limits=d$Features[1:topn])+
      labs(title = paste0("Feature Importance of ",model$learner$name," Model"))+
      theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(angle = 45, hjust = 1,size = 12))
  }
  return(plot(p))
}


# compute the percent of positive response captured in the top ?% of predicted probabilities
#------------------------ example ------------------------
# xgb_mod = train(xgb_lrn, tsk.train)
# xgb_pred = predict(xgb_mod, tsk.test)
# cumulativeCapturedLift(xgb_pred$data,"prob.Y",0.4)
#---------------------------------------------------------
cumulativeCapturedLift = function(predictData, classname, percent) {
  predictData = arrange(predictData, by = desc(predictData[,paste0("prob.",classname)]))
  totalPositive = sum(predictData$truth==classname)
  predictData = predictData[1:floor(percent*nrow(predictData)),'truth']
  result = sum(predictData == classname)/totalPositive
  return(result)
}

