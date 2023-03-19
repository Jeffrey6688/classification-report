# NAME: Enze Zhang 001027423
# Data set used is bank_personal_loan
# Aim is to predict whether a customer can be "upsold" to a personal banking loan

# load the data
#setwd("~/Desktop/regression and classifcation")
loan <- read.csv('bank_personal_loan.csv')

# basic info and plots
library("skimr")
library("DataExplorer")
library('corrplot')
library("ggplot2")

skim(loan)
summary(loan)
plot_bar(loan, ncol = 3)
plot_histogram(loan, ncol = 4)
plot_boxplot(loan, by = "Personal.Loan", ncol = 4)

loan.cor <- cor(loan)
loan.cor[9,1:13] # Correlation of personal loan
corrplot(loan.cor, method="circle") 


###### Modelling
library("data.table")
library("mlr3verse")

set.seed(1111) # Set seed

loan$Personal.Loan <- as.factor(loan$Personal.Loan) # Turn target variable to factor

# Define task
loan_task <- TaskClassif$new(id = "bank_personal_loan", 
                             backend = loan, 
                             target = "Personal.Loan", 
                             positive = '1')

# Cross validation resampling strategy
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(loan_task)

# Feature selection
filter <- flt("information_gain")
filter$calculate(loan_task)
as.data.table(filter)

# Train test split
split <- partition(loan_task, ratio = 0.8)

###### Define base learners
# Logistic regression
set.seed(1111)
lrn_log_reg<- lrn("classif.log_reg", predict_type = "prob")
res_log_reg <- resample(loan_task, lrn_log_reg, cv5, store_models = TRUE)
res_log_reg$aggregate(list(msr("classif.ce"), 
                           msr("classif.acc"), 
                           msr("classif.auc"), 
                           msr("classif.fpr"), 
                           msr("classif.fnr")))
#classif.ce classif.acc classif.auc classif.fpr classif.fnr 
#0.04960000  0.95040000  0.95764052  0.01547785  0.37084893 


# SVM
set.seed(1111)
lrn_svm<- lrn("classif.svm", predict_type = "prob")
res_svm <- resample(loan_task, lrn_svm, cv5, store_models = TRUE)
res_svm$aggregate(list(msr("classif.ce"), 
                       msr("classif.acc"), 
                       msr("classif.auc"), 
                       msr("classif.fpr"), 
                       msr("classif.fnr")))
#classif.ce classif.acc classif.auc classif.fpr classif.fnr 
#0.02300000  0.97700000  0.98536511  0.00774289  0.16631393 


# Tree
set.seed(1111)
lrn_tree_1 <- lrn("classif.rpart", predict_type = "prob", xval=10)
res_tree_1 <- resample(loan_task, lrn_tree_1, cv5, store_models = TRUE)
res_tree_1$aggregate(list(msr("classif.ce"), 
                          msr("classif.acc"), 
                          msr("classif.auc"), 
                          msr("classif.fpr"), 
                          msr("classif.fnr")))
#classif.ce classif.acc classif.auc classif.fpr classif.fnr 
#0.017000000 0.983000000 0.968606221 0.006400505 0.116057322 


# Random forest: initial model
set.seed(111)
lrn_ranger_1 <- lrn("classif.ranger", predict_type = "prob")
res_ranger_1 <- resample(loan_task, lrn_ranger_1, cv5, store_models = TRUE)
res_ranger_1$aggregate(list(msr("classif.ce"), 
                            msr("classif.acc"), 
                            msr("classif.auc"), 
                            msr("classif.fpr"), 
                            msr("classif.fnr")))
#classif.ce classif.acc classif.auc classif.fpr classif.fnr 
#0.013600000 0.986400000 0.996564737 0.002208884 0.120983878 

lrn_ranger_1$train(loan_task, split$train)
lrn_ranger_1$model$prediction.error


# hyperparameter selection or tuning by using grid search
set.seed(1111)
lrn_ranger <- lrn("classif.ranger", 
                  predict_type = "prob", 
                  mtry = to_tune(3,12),
                  num.trees = to_tune(100,1000),
                  max.depth = to_tune(5,15)
                  )

instance2 = tune(
  method = tnr("grid_search"),
  task = loan_task,
  learner = lrn_ranger,
  resampling = cv5,
  measures = list(msr("classif.ce"), 
                  msr("classif.acc"), 
                  msr("classif.auc"), 
                  msr("classif.fpr"), 
                  msr("classif.fnr")),
  terminator = trm("run_time", secs = 120)
)
instance2$result # By using grid search, mtry = 9 & num.trees = 200 has lowest error

# Random forest improved model
set.seed(111)
lrn_ranger_2 <- lrn("classif.ranger", 
                    predict_type = "prob", 
                    mtry = 6, 
                    num.trees = 500,
                    max.depth = 12
                    )
res_ranger_2 <- resample(loan_task, lrn_ranger_2, cv5, store_models = TRUE)
res_ranger_2$aggregate(list(msr("classif.ce"), 
                            msr("classif.acc"), 
                            msr("classif.auc"), 
                            msr("classif.fpr"), 
                            msr("classif.fnr")))
#classif.ce classif.acc classif.auc classif.fpr classif.fnr 
#0.012800000 0.987200000 0.997413515 0.003096423 0.104854749 

lrn_ranger_2$train(loan_task, split$train)
pred_ranger <- lrn_ranger_2$predict(loan_task, split$test)
pred_ranger$confusion
measure = list(msr("classif.ce"), 
               msr("classif.acc"), 
               msr("classif.auc"), 
               msr("classif.fpr"), 
               msr("classif.fnr"),
               )
pred_ranger$score(measure)
#classif.ce classif.acc classif.auc classif.fpr classif.fnr 
#0.011000000 0.989000000 0.997614768 0.002212389 0.093750000 


# Model performance plots
lrn_ranger_2$model$prediction.error #OOB prediction error 0.01
object = resample(loan_task, lrn_ranger_2, cv5)
autoplot(object, type = "roc") # ROC curve
autoplot(object, type = "prc")
autoplot( pred_ranger)

true <- as.numeric(as.character(pred_ranger$truth))
pred <- as.numeric(as.character(pred_ranger$prob))
df <- data.frame(true, pred)
library(caret)
calPlotData <- calibration(pred_ranger$truth ~ pred, data = df)
xyplot(calPlotData, auto.key = list(columns =2))


