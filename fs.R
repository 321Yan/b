# library(ggplot2)
# library(dpylr)

plotFeatureImportance = function(model,topn = NULL) {
  f = getFeatureImportance(model)
  imp = f$res
  d = data.frame(Features = names(imp), Importance = as.numeric(imp))
  d = arrange(d,desc(Importance))
  if (is.null(topn)) {
    p = ggplot(d,aes(Features,Importance))+geom_col()+
      scale_x_discrete(limits=d$Features)+
      labs(title = paste0("Feature Importance of ",model$learner$name," Model"))+
      theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(angle = 45, hjust = 1))
  } else {
    p = ggplot(d[1:topn,],aes(Features,Importance))+geom_col()+
      scale_x_discrete(limits=d$Features[1:topn])+
      labs(title = paste0("Feature Importance of ",model$learner$name," Model"))+
      theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(angle = 45, hjust = 1,size = 12))
  }
  return(plot(p))
}


# r = resample(rf_lrn, tsk, resampling = rdesc, show.info = T, models = T,measures = mae)
# plotFeatureImportance(r$models[[1]],10)