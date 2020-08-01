#=-=-=-=
library(e1071)
library(caret)
library(ROSE)
library(doParallel)
#=-=-=-=

#Converting to R DataFrame Objects
training_df <- as.data.frame(training_data)
test_df <- as.data.frame(test_data)

#Splitting the training data into Training (75%) and Validation (25%)
set.seed(120)
ntrain <- round(nrow(training_df)*0.75)  # 75% for training set
index <- sample(nrow(training_df),ntrain)

training_x <- training_df[index,]
training_x$BF_AreaShape_Area <- as.numeric(training_x$BF_AreaShape_Area)
training_x$class <- factor(training_x$class)

validation_x <- training_df[-index,]
validation_x$BF_AreaShape_Area <- as.numeric(validation_x$BF_AreaShape_Area)
validation_x$class <- factor(validation_x$class)

########################
# SVM - BASELINE MODEL #
########################
#BASELINE SVM - ALL CLASSES - ALL VARIABLES - NO ROSE - LINEAR KERNEL - COST = 1
svm_baseline <- svm(x = training_x[,1:213], y = training_x[,214], type="C-classification", 
                    kernel = "linear", cost = 1)

#Prediction & Confusion Matrix on Training Data
pred_svm_baseline <- predict(svm_baseline, training_x[,1:213], decision.values = TRUE)
confusionMatrix(pred_svm_baseline, training_x[,214], dnn=c("Prediction", "Truth"))

#Prediction & Confusion Matrix on Validation Data
pred_svm_baseline_valid <- predict(svm_baseline, validation_x[,1:213], decision.values = TRUE)
confusionMatrix(pred_svm_baseline_valid, validation_x[,214], dnn=c("Prediction", "Truth"))
#confusion on Metaphase and Prophase

#############################
# REMOVE CORRELATED FEATURES#
#############################
# calculate correlation matrix
cor <- cor(training_x[,1:213])
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(cor, cutoff=0.75)
# print indexes of highly correlated attributes
print(highlyCorrelated)

training_x_noncorr <- training_x[,-highlyCorrelated]
validation_x_noncorr <- validation_x[,-highlyCorrelated]
#class variable is kept

#######################################
# SVM - NON-CORRELATED FEATURES MODEL #
#######################################
#BASELINE SVM - ALL CLASSES - NONCORR VARIABLES - NO ROSE - LINEAR KERNEL - COST = 1
svm_baseline_noncorr <- svm(x = training_x_noncorr[,1:80], y = training_x_noncorr[,81], 
                            type="C-classification", kernel = "linear", cost = 1)

#Prediction & Confusion Matrix on Training Data
pred_svm_baseline_noncorr <- predict(svm_baseline_noncorr, training_x_noncorr[,1:80], 
                                     decision.values = TRUE)
confusionMatrix(pred_svm_baseline_noncorr, training_x_noncorr[,81], dnn=c("Prediction", "Truth"))

#Prediction & Confusion Matrix on Validation Data
pred_svm_baseline_noncorr_valid <- predict(svm_baseline_noncorr, validation_x[,1:80],
                                           decision.values = TRUE)
confusionMatrix(pred_svm_baseline_noncorr_valid, validation_x_noncorr[,81], dnn=c("Prediction", "Truth"))
#extremely weak performance for metaphase, prophase also misclassified as interphase more
#kappa decreased

#########
# SMOTE #
#########
#ROSE package is used here to have full control over balancing.
#ROSE only allows binary classification, so for each step this classification is 'reset'.

#STEP 1 - Oversample Anaphase
training_x_noncorr$binary <- 0
training_x_noncorr$binary[training_x_noncorr$class == "Anaphase"] <- 1
training_x_noncorr$binary <- as.factor(training_x_noncorr$binary)

rose_under_5 <- ovun.sample(binary ~ ., data = training_x_noncorr, 
                          method = "over",N = 22670)$data

#STEP 2 - Oversample Prophase (396 --> 401)
rose_under_5$binary <- 0
rose_under_5$binary[rose_under_5$class == "Prophase"] <- 1

rose_part_5 <- rose_under_5[rose_under_5$class != "Anaphase",]
rose_part_5_2 <- rose_part_5[rose_part_5$class != "Telophase",]
rose_part_5_3 <- rose_part_5_2[rose_part_5_2$class != "Interphase",]
rose_part_5_3$binary <- as.factor(rose_part_5_3$binary)

rose_under_5_step2 <- ovun.sample(binary ~ ., data = rose_part_5_3, 
                                method = "under",N = 451)$data

#STEP 3 - Oversample Metaphase (47 --> 6750)
rose_under_5_step2$binary <- 0
rose_under_5_step2$binary[rose_under_5_step2$class == "Metaphase"] <- 1

rose_part2_5 <- rose_under_5_step2[rose_under_5_step2$class != "Anaphase",]
rose_part2_5_2 <- rose_part2_5[rose_part2_5$class != "Telophase",]
rose_part2_5_3 <- rose_part2_5_2[rose_part2_5_2$class != "Interphase",]

rose_under_5_step3 <- ovun.sample(binary ~ ., data = rose_part2_5_3, 
                                  method = "over",N = 7151)$data

# STEP 4 - Undersample Interphase (21302 --> 3092)
rose_under_5_step2$binary <- 0
rose_under_5_step2$binary[rose_under_5_step2$class == "Interphase"] <- 1

rose_part3_5 <- rose_under_5[rose_under_5$class != "Anaphase",]
rose_part3_5_2 <- rose_part3_5[rose_part3_5$class != "Telophase",]
rose_part3_5_3 <- rose_part3_5_2[rose_part3_5_2$class != "Prophase",]
rose_part3_5_3$binary <- as.factor(rose_part3_5_3$binary)

rose_under_5_step4 <- ovun.sample(class ~ ., data = rose_part3_5_3, 
                                  method = "under",N = 3142)$data

#Combine oversampled/undersampled records into one R object
rose_partA_5 <- rose_under_5[rose_under_5$class != "Prophase",]
rose_partA2_5 <- rose_partA_5[rose_partA_5$class != "Interphase",]
rose_part1_5 <- rose_partA2_5[rose_partA2_5$class != "Metaphase",]
rose_partB_5 <- rose_under_5_step2[rose_under_5_step2$class != "Metaphase",]
rose_partC_5 <- rose_under_5_step4[rose_under_5_step4$class != "Metaphase",]
rose_partD_5 <- rose_under_5_step3[rose_under_5_step3$class != "Prophase",]

rose_final_5 <- rbind(rose_part1_5, rose_partB_5, rose_partC_5, rose_partD_5)

#Newly 'balanced' observation set
table(rose_final_5$class)

#Cleaning up R workspace
rm(rose_final_5, rose_part_5, rose_part_5_2, rose_part_5_3, rose_part1_5, rose_part2_5, rose_part2_5_2,
   rose_part2_5_3, rose_part3_5, rose_part3_5_2, rose_part3_5_3, rose_partA_5, rose_partA2_5, 
   rose_partB_5,rose_partC_5, rose_partD_5, rose_under_5, rose_under_5_step2, rose_under_5_step3,
   rose_under_5_step4)

#Splitting predictor features to X and classification label to Y
rose_final_y <- rose_final_5$class
rose_final_x <- rose_final_5

#Predictors X should not have 'class' nor 'binary' (latter created for SMOTE)
rose_final_x$class <- NULL
rose_final_x$binary <- NULL

#Neither should the vector 'binary' be used again for modeling purposes
rose_final_5$binary <- NULL

###################################
# SVM - SMOTE BALANCED DATA MODEL #
###################################
set.seed(100)
svm_rose <- svm(x = rose_final_x, y = rose_final_y, type="C-classification", 
                                kernel = "linear", cost = 1)

#Prediction & Confusion Matrix on Training Data
pred_svm_rose <- predict(svm_rose, training_x_noncorr[,1:80], decision.values = TRUE)
confusionMatrix(pred_svm_rose, training_x_noncorr[,81], dnn=c("Prediction", "Truth"))

#Prediction & Confusion Matrix on Validation Data
pred_svm_rose_valid <- predict(svm_rose, validation_x_noncorr[,1:80], decision.values = TRUE)
confusionMatrix(pred_svm_rose_valid, validation_x_noncorr[,81], dnn=c("Prediction", "Truth"))

###########################################
#RFE - DIMENSION REDUCTION METHOD (CARET) #
###########################################
#Only 3-fold CV is used for limitations to computational power
control <- rfeControl(functions = caretFuncs, method="cv", number=3)

#Some data prep is required to let RFE run without errors
#This primarily includes removing near-zero variance features
x2 <- model.matrix(~., data = rose_final_x)[,-1]
nzv <- nearZeroVar(x2)
x3 <- x2[, -nzv]
x4_rfe <- as.data.frame(x3)
rm(x2, x3)

#RFE is computationally heavy. The Caret package in R can also be very slow in some occasions.
# A package is used to let RFE run in parallel using multiple cores (package doParallel)
cl <- makeCluster(detectCores(), type='PSOCK')
registerDoParallel(cl)

#For classificaiton, you can run RFE to optimize either the metric 'Accuracy' or 'Kappa'
#RFE to optimize Kappa
results_kappa <- rfe(x4_rfe, rose_final_y, sizes=c(1:78), rfeControl=control, method = "svmLinear",
               metric = "Kappa", preProc = c("center", "scale"))
predictors(results_kappa)
#This shows a very useful plot: number of most contributing features for each x number of features
#allowed vs. the Kappa score
plot(results_kappa, type=c("g", "o"))

#RFE to optimize Accuracy
results_acc <- rfe(x4_rfe, rose_final_y, sizes=c(1:78), rfeControl=control, method = "svmLinear",
                     metric = "Accuracy", preProc = c("center", "scale"))
predictors(results_acc)
#This shows a very useful plot: number of most contributing features for each x number of features
#allowed vs. the Accuracy score
plot(results_acc, type=c("g", "o"))
#plots of RFE using Kappa and Accuracy are similar

#The following two blocks of code look very similar.
#They are used to fetch the relevant features for each number X of features selected by RFE to optimize
#the given metric.

#To fetch the features for each model of X variables to optimize Kappa
variables_kappa <- results_kappa$variables
#For the following, the number in the index filtering represents the number of features you want
#We use 34 here to have significant feature reduction while retaining good Kappa scores
variables_kappa_numvar <- variables_kappa[variables_kappa$Variables==34,]
#fold 1, 2, 3 are available, dimensions may differ between folds. Good to compare them.
variables_kappa_numvar_fold1 <- variables_kappa_numvar[variables_kappa_numvar$Resample == "Fold1",]
variables_kappa_numvar_fold2 <- variables_kappa_numvar[variables_kappa_numvar$Resample == "Fold2",]
variables_kappa_numvar_fold3 <- variables_kappa_numvar[variables_kappa_numvar$Resample == "Fold3",]
rm(variables_kappa, variables_kappa_numvar)
variables_kappa_numvar_fold1$var

#Converting rownames so selecting features on Training and Validation data will be made easier.
rownames(variables_kappa_numvar_fold1) <- variables_kappa_numvar_fold1$var
rownames(variables_kappa_numvar_fold2) <- variables_kappa_numvar_fold2$var
rownames(variables_kappa_numvar_fold3) <- variables_kappa_numvar_fold3$var

#Filtering datasets with relevant features
rose_final_x_kappa <- rose_final_x[,rownames(variables_kappa_numvar_fold1)]
training_x_kappa <- training_x[,rownames(variables_kappa_numvar_fold1)]
validation_x_kappa <- validation_x[,rownames(variables_kappa_numvar_fold1)]

#re-add class vector
training_x_kappa$class <- training_x$class
validation_x_kappa$class <- validation_x$class


#To fetch the features for each model of X variables to optimize Accuracy
variables_acc <- results_acc$variables
#For the following, the number in the index filtering represents the number of features you want
#We use 35 here to have significant feature reduction while retaining good Accuracy scores
variables_acc_numvar <- variables_acc[variables_acc$Variables==35,]
#fold 1, 2, 3 are available, dimensions may differ between folds. Good to compare them.
variables_acc_numvar_fold1 <- variables_acc_numvar[variables_acc_numvar$Resample == "Fold1",]
variables_acc_numvar_fold2 <- variables_acc_numvar[variables_acc_numvar$Resample == "Fold2",]
variables_acc_numvar_fold3 <- variables_acc_numvar[variables_acc_numvar$Resample == "Fold3",]
rm(variables_acc, variables_acc_numvar)
variables_acc_numvar_fold1$var

#Converting rownames so selecting features on Training and Validation data will be made easier.
rownames(variables_acc_numvar_fold1) <- variables_acc_numvar_fold1$var
rownames(variables_acc_numvar_fold2) <- variables_acc_numvar_fold2$var
rownames(variables_acc_numvar_fold3) <- variables_acc_numvar_fold3$var

#Filtering datasets with relevant features
rose_final_x_acc <- rose_final_x[,rownames(variables_acc_numvar_fold1)]
training_x_acc <- training_x[,rownames(variables_acc_numvar_fold1)]
validation_x_acc <- validation_x[,rownames(variables_acc_numvar_fold1)]

#re-add class vector
training_x_acc$class <- training_x$class
validation_x_acc$class <- validation_x$class

######################################################
# SVM - USING SMOTE BALANCED DATA, ACCURACY FEATURES #
######################################################
set.seed(100)
svm_rose_acc <- svm(x = rose_final_x_acc, y = rose_final_y, type="C-classification", 
                    kernel = "linear", cost = 1)

#Prediction & Confusion Matrix on Training Data
pred_svm_rose_acc <- predict(svm_rose_acc, training_x_acc[,1:35], decision.values = TRUE)
confusionMatrix(pred_svm_rose_acc, training_x_acc[,36], dnn=c("Prediction", "Truth"))

#Prediction & Confusion Matrix on Validation Data
pred_svm_rose_acc_valid <- predict(svm_rose_acc, validation_x_acc[,1:35], decision.values = TRUE)
confusionMatrix(pred_svm_rose_acc_valid, validation_x_acc[,36], dnn=c("Prediction", "Truth"))
#with 35 variables, only accuracy for prophase has deteriorated significantly

#################################################################
# SVM - FINAL MODEL - USING SMOTE BALANCED DATA, KAPPA FEATURES #
#################################################################
set.seed(100)
svm_rose_kappa <- svm(x = rose_final_x_kappa, y = rose_final_y, type="C-classification", 
                      kernel = "linear", cost = 1)

#Prediction & Confusion Matrix on Training Data
pred_svm_rose_kappa <- predict(svm_rose_kappa, training_x_kappa[,1:34], decision.values = TRUE)
confusionMatrix(pred_svm_rose_kappa, training_x_kappa[,35], dnn=c("Prediction", "Truth"))

#Prediction & Confusion Matrix on Validation Data
pred_svm_rose_kappa_valid <- predict(svm_rose_kappa, validation_x_kappa[,1:34], decision.values = TRUE)
confusionMatrix(pred_svm_rose_kappa_valid, validation_x_kappa[,35], dnn=c("Prediction", "Truth"))
#with 34 variables, only accuracy for prophase has deteriorated significantly