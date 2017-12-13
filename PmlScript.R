#######################################################################################
### ML FINAL PROJECT ###

# pml<- read.csv("pml-training.csv") # works if file is in wd, otherwise see below
# View(pml)
library(RCurl)
myfile<-getURL('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv',
               ssl.verifyhost=FALSE, ssl.verifypeer=FALSE)
pml<- read.csv(textConnection(myfile), header = TRUE)
View(pml)
pml2<- pml[,colSums(is.na(pml))==0]
pml2<- pml2[,!apply(pml2, 2, function(x) any(x==""))]
View(pml2)
pml2<-pml2[,8:60]

library(caret)
control1 <- rfeControl(functions=rfFuncs, method="cv", number=4)
control2 <- rfeControl(functions=treebagFuncs, method="cv", number=4)
control3<- rfeControl(functions = ldaFuncs, method = "cv", number = 4)

# Now, we time and carry out recursive feature elimination for our models
system.time(
  results1 <- rfe(pml2[,names(pml2)!="classe"], pml2[,"classe"],
                sizes=c(1:15), rfeControl=control1)
)
system.time(
  results2 <- rfe(pml2[,names(pml2)!="classe"], pml2[,"classe"],
                sizes=c(1:15), rfeControl=control2)
)
system.time(
  results3 <- rfe(pml2[,names(pml2)!="classe"], pml2[,"classe"],
                sizes=c(1:15), rfeControl=control3)
)
#results4 <- rfe(pml2[,names(pml2)!="classe"], pml2[,"classe"],
#sizes=c(1:15), rfeControl=control4)  #WAY too slow

# Next, we look at 
predictors(results1) # in order of importance
plot(results1, type=c("g", "o")) # how many features do we need? (13-14+)

predictors(results2)
plot(results2, type=c("g", "o")) # 14+ features looks good

predictors(results3)
plot(results3, type=c("g", "o")) # all features give ~ 0.7 accuracy :(

# Now we train some models to assess in-sample accuracy -
# using the features we selected above 
control <- trainControl(method="cv", number=5) #using 5-fold cross validation
bench1 <- train(classe~.,
                data=pml2[,c(predictors(results1)[1:14],"classe")],
                method="rf", preProcess=c("center", "scale"),
                trControl=control)
importance1 <- varImp(bench1, scale=FALSE)
print(importance1)
plot(importance1) #compare variable importance visually
a1<-predict(bench1, newdata = pml2[,names(pml2)!="classe"])
mean(a1==pml2[,"classe"]) # rough in-sample accuracy ( ~ 0.999 )
confusionMatrix(data = a1, reference = pml2$classe) #view by class accuracy

bench2 <- train(classe~.,
                data=pml2[,c(predictors(results2)[1:14],"classe")],
                method="treebag", preProcess=c("center", "scale"),
                trControl=control)
importance2 <- varImp(bench2, scale=FALSE)
print(importance2)
plot(importance2)
a2<-predict(bench2, newdata = pml2[,names(pml2)!="classe"])
mean(a2==pml2[,"classe"]) # ~ 0.999
confusionMatrix(data = a2, reference = pml2$classe)

bench3 <- train(classe~.,
                data=pml2[,c(predictors(results3),"classe")],
                method="lda", preProcess=c("center", "scale"),
                trControl=control)
importance3 <- varImp(bench3, scale=FALSE)
print(importance3)
plot(importance3)
a3<-predict(bench3, newdata = pml2[,names(pml2)!="classe"])
mean(a3==pml2[,"classe"]) # ~ 0.705
confusionMatrix(data = a3, reference = pml2$classe)

bench4 <- train(classe~.,
               data=pml2[,c(predictors(results2)[1:15],"classe")],
               method="gbm", preProcess=c("center", "scale"),
               trControl=control)
importance4 <- varImp(bench4, scale=FALSE)
print(importance4)
plot(importance4)
a4<-predict(bench4, newdata = pml2[,names(pml2)!="classe"])
mean(a4==pml2[,"classe"]) # ~ 0.961
confusionMatrix(data = a4, reference = pml2$classe)

# Here we create 3 data sets from our training data:
# namely training, validation, and testing sets
inTrain<-createDataPartition(pml2$classe, times = 1, p=0.80)[[1]]
TrainSet<-pml2[inTrain,]
ValidSet0<-pml2[-inTrain,]
innValid<-createDataPartition(ValidSet0$classe, times = 1, p=0.5)[[1]]
ValidSet<-ValidSet0[innValid,]
TestSet<-ValidSet0[-innValid,]
any(ValidSet %in% TrainSet) # check for overlap, there is none
any(TestSet %in% TrainSet) # same as above
any(TestSet %in% ValidSet) # same as above

control <- trainControl(method="cv", number=5) # again, we use 5-fold CV

# now we train new models on the 'new' training set and we test for
# something akin to out of sample accuracy using the validation set
model1 <- train(classe~.,
                data=TrainSet[,c(predictors(results1)[1:14],"classe")],
                method="rf", preProcess=c("center", "scale"),
                trControl=control)
# Here we make predictions on the validation set using the model
# we just trained on the training set
ValidSet$rf_preds<-predict(model1,
                           newdata = ValidSet[,names(ValidSet)!="classe"])
mean(ValidSet$rf_preds==ValidSet$classe) # akin to out of sample accuracy (0.990)
confusionMatrix(ValidSet$rf_preds,ValidSet$classe)
# for stacking, we create predictions from the 'test' set
TestSet$rf_preds<-predict(model1,
                             newdata = TestSet[,names(TestSet)!="classe"])

model2 <- train(classe~.,
                data=TrainSet[,c(predictors(results2)[1:14],"classe")],
                method="treebag", preProcess=c("center", "scale"),
                trControl=control)
ValidSet$treebag_preds<-predict(model2,
                                newdata = ValidSet[,names(ValidSet)!="classe"])
mean(ValidSet$treebag_preds==ValidSet$classe) # akin to out of sample accuracy (.985)
confusionMatrix(ValidSet$treebag_preds,ValidSet$classe)
TestSet$treebag_preds<-predict(model2,
                                  newdata = TestSet[,names(TestSet)!="classe"])

model3 <- train(classe~.,
                data=TrainSet[,c(predictors(results3),"classe")],
                method="lda", preProcess=c("center", "scale"),
                trControl=control)
ValidSet$lda_preds<-predict(model3,
                            newdata = ValidSet[,names(ValidSet)!="classe"])
mean(ValidSet$lda_preds==ValidSet$classe) # akin to out of sample accuracy (.698)
confusionMatrix(ValidSet$lda_preds,ValidSet$classe)
TestSet$lda_preds<-predict(model3,
                              newdata = TestSet[,names(TestSet)!="classe"])

model4 <- train(classe~.,
                data=TrainSet[,c(predictors(results2)[1:20],"classe")],
                method="gbm", preProcess=c("center", "scale"),
                trControl=control)
ValidSet$gbm_preds<-predict(model4,
                            newdata = ValidSet[,names(ValidSet)!="classe"])
mean(ValidSet$gbm_preds==ValidSet$classe) # akin to out of sample accuracy (.971)
confusionMatrix(ValidSet$gbm_preds,ValidSet$classe)
TestSet$gbm_preds<-predict(model4,
                              newdata = TestSet[,names(TestSet)!="classe"])

# Next we create a dataframe with just the validation set predictions
# from all of the models
PredsDF_V<-data.frame(rfPreds = ValidSet$rf_preds,
                    treebagPreds = ValidSet$treebag_preds,
                    #ldaPreds = ValidSet$lda_preds,
                    gbmPreds = ValidSet$gbm_preds,
                    classe = ValidSet$classe)

# We train a model on the predictions in the dataframe populated
# with only predictions based on the validation set
StackedMod<- train(classe~.,data = PredsDF_V, method = "rf",
                   preProcess=c("center", "scale"),
                   trControl = control)

# We create a dataframe with just the 'test' set predictions
# made by applying the models trained on the training set
PredsDF_T<-data.frame(rfPreds = TestSet$rf_preds,
                      treebagPreds = TestSet$treebag_preds,
                      #ldaPreds = TestSet$lda_preds,
                      gbmPreds = TestSet$gbm_preds,
                      classe = TestSet$classe)
# We make predictions using the stacked model on the 'test' set
# predictions, which were made by the models trained on the training
# set only
TestPreds<- predict(StackedMod, PredsDF_T)
# better assessment of the out of sample accuracy (0.986)
mean(TestPreds==TestSet$classe)
confusionMatrix(TestPreds,TestSet$classe)

#compare to individual model accuracy
mean(PredsDF_T$rfPreds==PredsDF_T$classe)
mean(PredsDF_T$treebagPreds==PredsDF_T$classe)
mean(PredsDF_T$gbmPreds==PredsDF_T$classe)

# simple majority vote, first define Mode fx
Mode <- function(x) {
ux <- unique(x)
ux[which.max(tabulate(match(x, ux)))]
}

for(i in 1:nrow(PredsDF_T)){
  PredsDF_T$maj[i]<-as.character(Mode(PredsDF_T[i,1:3])[[1]])
}

mean(PredsDF_T$classe==PredsDF_T$maj) #(0.984), boo!


#######################################################################################
## Here, we're going to try blending the top two models and see if we can't
## get a bump in our proxy for out of sample error

# this first code chunk verifies that ~14+ is a good number of predictors for rf
result<-rfcv(pml2[,names(pml2)!="classe"],pml2$classe, cv.fold = 10)
with(result, plot(n.var, error.cv, log="x", type="o", lwd=2))
# I'm selecting the 18 features that occur in the top 20 features previously
# determined for each of the top two models
intersect(predictors(results1)[1:20],predictors(results2)[1:20])
feats<-intersect(predictors(results1)[1:20],predictors(results2)[1:20])
pml3<-pml2[,c(feats,"classe")] # new dataset that contains only those features
# now we cutthe training set into three exclusive random folds
inTrain<-createDataPartition(pml3$classe, times = 1, p=0.80)[[1]]
TrainSet<-pml3[inTrain,]
ValidSet0<-pml3[-inTrain,]
innValid<-createDataPartition(ValidSet0$classe, times = 1, p=0.5)[[1]]
ValidSet<-ValidSet0[innValid,]
TestSet<-ValidSet0[-innValid,]
any(ValidSet %in% TrainSet) # check for overlap, there is none
any(TestSet %in% TrainSet) # same as above
any(TestSet %in% ValidSet) # same as above

control <- trainControl(method="cv", number=10) # here we'll use 10-fold CV

# now we train new models on the new 'new' training set and we test for
# something akin to out of sample accuracy using the validation set
model1 <- train(classe~.,
                data=TrainSet,
                method="rf", preProcess=c("center","scale"),
                trControl=control)
# Here we make predictions on the validation set using the model
# we just trained on the training set so we can estimate out of
# sample error
ValidSet$rf_preds<-predict(model1,
                           newdata = ValidSet[,feats])
mean(ValidSet$rf_preds==ValidSet$classe) # akin to out of sample accuracy (0.990)
confusionMatrix(ValidSet$rf_preds,ValidSet$classe)
# for blending, we create predictions from the 'test' set to be added as
# new column(s) to the 'test' set
TestSet$rf_preds<-predict(model1,
                          newdata = TestSet[,feats])

model2 <- train(classe~.,
                data=TrainSet,
                method="treebag", preProcess=c("center","scale"),
                trControl=control)
ValidSet$treebag_preds<-predict(model2,
                                newdata = ValidSet[,feats])
mean(ValidSet$treebag_preds==ValidSet$classe) # akin to out of sample accuracy (.985)
confusionMatrix(ValidSet$treebag_preds,ValidSet$classe)
TestSet$treebag_preds<-predict(model2,
                               newdata = TestSet[,feats])
# Here we train the blended model on the 'validation' set which we've added
# columns of predictions to from the two component models
BlendMod<- train(classe~., data = ValidSet, method = "rf",
                   preProcess =c("center","scale"),
                   trControl = control)
BlendedPreds<- predict(BlendMod, TestSet[,names(TestSet)!="classe"])
mean(BlendedPreds==TestSet$classe) # (~0.994) akin to out of sample accuracy for our
# final, blended model (better than either model alone, which was not the case when 
# we trained a stacked model on the predictions from the other initial models alone)

#######################################################################################
##### Now, we'll predict on the actual testing set ##### 
setwd("~/R_wd")
realTestDF<-read.csv('pml-testing.csv') # see top of script for alternate method
realTestDF<-realTestDF[,colSums(is.na(realTestDF))==0]
realTestDF<-realTestDF[,!apply(realTestDF, 2, function(x) any(x==""))]
realTestDF<-realTestDF[,8:60]
realTestDF$rf_preds<-predict(model1,
                          newdata = realTestDF[,feats])
realTestDF$treebag_preds<-predict(model2,
                               newdata = realTestDF[,feats])
FinalPreds<- predict(BlendMod, realTestDF[,names(realTestDF)!="classe"])
print(FinalPreds)
realTestDF$final_preds<-FinalPreds
save('realTestDF', file = 'realTestDF.rda')
write.csv(realTestDF, 'realTestDF.csv', row.names = FALSE)
realTestDF2<-data.frame(realTestDF[,feats],realTestDF[,((ncol(realTestDF)-2):ncol(realTestDF))])
save('realTestDF2', file = 'realTestDF2.rda')
write.csv(realTestDF2, 'realTestDF2.csv', row.names = FALSE)
save('FinalPreds', file = 'FinalPreds.rda')
save.image('mlFinalProjWS.RData')
#########################################################
#######################################################################################
