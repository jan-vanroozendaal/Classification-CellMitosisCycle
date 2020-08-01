#=-=-=-=-=-=-=
library(caret)
library(ROSE)
library(doParallel)
#=-=-=-=-=-=-=

#Converting to R DataFrame Objects
training_df <- as.data.frame(training_data)
test_df <- as.data.frame(test_data)

#Splitting the training data into Training (75%) and Validation (25%)
set.seed(200)
ntrain <- round(nrow(training_df)*0.75)  # 75% for training set
index <- sample(nrow(training_df),ntrain)

training_x <- training_df[index,]
training_x$BF_AreaShape_Area <- as.numeric(training_x$BF_AreaShape_Area)

validation_x <- training_df[-index,]
validation_x$BF_AreaShape_Area <- as.numeric(validation_x$BF_AreaShape_Area)

training_y <- training_df$class[index]
validation_y <- training_df$class[-index]

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

#########
# SMOTE #
#########
#ROSE package is used here to have full control over balancing.
#ROSE only allows binary classification, so for each step this classification is 'reset'.

#STEP 1 - Oversample Anaphase (7 --> 1207)
training_x_noncorr$binary <- 0
training_x_noncorr$binary[training_x_noncorr$class == "Anaphase"] <- 1
training_x_noncorr$binary <- as.factor(training_x_noncorr$binary)

rose_under_5 <- ovun.sample(binary ~ ., data = training_x_noncorr, 
                            method = "over",N = 22970)$data

#STEP 2 - Oversample Prophase (396 --> 411)
rose_under_5$binary <- 0
rose_under_5$binary[rose_under_5$class == "Prophase"] <- 1
rose_part_5 <- rose_under_5[rose_under_5$class != "Anaphase",]
rose_part_5_2 <- rose_part_5[rose_part_5$class != "Telophase",]
rose_part_5_3 <- rose_part_5_2[rose_part_5_2$class != "Interphase",]
rose_part_5_3$binary <- as.factor(rose_part_5_3$binary)

rose_under_5_step2 <- ovun.sample(binary ~ ., data = rose_part_5_3, 
                                  method = "under",N = 459)$data

#STEP 3 - Oversample Metaphase (47 --> 1148)
rose_under_5_step2$binary <- 0
rose_under_5_step2$binary[rose_under_5_step2$class == "Metaphase"] <- 1
rose_part2_5 <- rose_under_5_step2[rose_under_5_step2$class != "Anaphase",]
rose_part2_5_2 <- rose_part2_5[rose_part2_5$class != "Telophase",]
rose_part2_5_3 <- rose_part2_5_2[rose_part2_5_2$class != "Interphase",]

rose_under_5_step3 <- ovun.sample(binary ~ ., data = rose_part2_5_3, 
                                  method = "over",N = 1559)$data

# STEP 4 - Undersample Interphase (21302 --> 1494)
rose_under_5_step2$binary <- 0
rose_under_5_step2$binary[rose_under_5_step2$class == "Interphase"] <- 1
rose_part3_5 <- rose_under_5[rose_under_5$class != "Anaphase",]
rose_part3_5_2 <- rose_part3_5[rose_part3_5$class != "Telophase",]
rose_part3_5_3 <- rose_part3_5_2[rose_part3_5_2$class != "Prophase",]
rose_part3_5_3$binary <- as.factor(rose_part3_5_3$binary)

rose_under_5_step4 <- ovun.sample(class ~ ., data = rose_part3_5_3, 
                                  method = "under",N = 1542)$data

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

############################################
# KNN - BASELINE - NON-CORRELATED FEATURES #
############################################
rose_final_x$class <- rose_final_y

#Repeated 10-fold cross-validation enabled
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

set.seed(100)
knn_model <- train(class~., data=rose_final_x, method = "knn", tuneLength = 20, trControl = ctrl,
                   preProc = c("center", "scale"))
knn_model #to understand which value of 'k' is used

#Prediction & Confusion Matrix on Training Data
pred <- predict(knn_model, newdata = training_x_noncorr)
confusionMatrix(pred, training_y)

#Prediction & Confusion Matrix on Validation Data
pred_valid <- predict(knn_model, newdata = validation_x_noncorr)
confusionMatrix(pred_valid, validation_y)

####################################################
# FEATURE SELECTION - CARET EMBEDDED FUNCTIONALITY #
####################################################
varImp(knn_model)
varImp <- as.data.frame(varImp(knn_model)$importance)
#Crate max of each row in new column 'max'
varImp[, "max"] <- apply(varImp[, 1:5], 1, max)
#sort table by max importance of each row
varImp <- varImp[order(varImp$max, decreasing = TRUE),]
varImp_filter <- varImp[varImp$max >= 20,] #change this number to your liking, lower means more features
#filter rose dataset with related variables
rose_final_x_varImp <- rose_final_x[,rownames(varImp_filter)]

#add class vector back to varImp datasets
rose_final_x_varImp$class <- rose_final_x$class

#prepping training and validation datasets with new importance variables
training_x_varImp <- training_x[,rownames(varImp_filter)]
validation_x_varImp <- validation_x[,rownames(varImp_filter)]


###########################################
# KNN - 'VARIMP' CARET IMPORTANT FEATURES #
###########################################
knn_model_varImp <- train(class~., data=rose_final_x_varImp, method = "knn", tuneLength = 20, 
                          trControl = ctrl, preProc = c("center", "scale"))

#Prediction & Confusion Matrix on Training Data
pred_varImp <- predict(knn_model_varImp, newdata = training_x_varImp)
confusionMatrix(pred_varImp, training_y)

#Prediction & Confusion Matrix on Validation Data
pred_varImp_valid <- predict(knn_model_varImp, newdata = validation_x_varImp)
confusionMatrix(pred_varImp_valid, validation_y)
#it seems even a slight reduction of variables has an impact om prformance (Metaphase, Prophase)

##############################################
#ALTERNATIVE FEATURE REDUCTION METTHOD - RFE #
##############################################
#Only 3-fold CV is used for limitations to computational power
ctrl_rfe <- rfeControl(method = "cv", number = 3)

#Some data prep is required to let RFE run without errors
#This primarily includes removing near-zero variance features
x2 <- model.matrix(~., data = rose_final_x)[,-1]
nzv <- nearZeroVar(x2)
x3 <- x2[, -nzv]
x4_rfe <- as.data.frame(x3)
rm(x2, x3)
x4_rfe$classInterphase <- NULL
x4_rfe$classMetaphase <- NULL
x4_rfe$classProphase <- NULL

#RFE is computationally heavy. The Caret package in R can also be very slow in some occasions.
# A package is used to let RFE run in parallel using multiple cores (package doParallel)
cl <- makeCluster(detectCores(), type='PSOCK')
registerDoParallel(cl)

#For classificaiton, you can run RFE to optimize either the metric 'Accuracy' or 'Kappa'
#RFE to optimize Kappa
results_kappa <- rfe(x4_rfe, rose_final_y, sizes=c(1:78), rfeControl=ctrl_rfe, method = "knn",
                     metric = "Kappa", tuneLength = 20, preProc = c("center", "scale"))
predictors(results_kappa)
#This shows a very useful plot: number of most contributing features for each x number of features
#allowed vs. the Kappa score
plot(results_kappa, type=c("g", "o"))

#RFE to optimize Accuracy
results_acc <- rfe(x4_rfe, rose_final_y, sizes=c(1:78), rfeControl=ctrl_rfe, method = "knn",
                   metric = "Accuracy", tuneLength = 20, preProc = c("center", "scale"))
predictors(results_acc)
#This shows a very useful plot: number of most contributing features for each x number of features
#allowed vs. the Accuracy score
plot(results_acc, type=c("g", "o"))

#The following two blocks of code look very similar.
#They are used to fetch the relevant features for each number X of features selected by RFE to optimize
#the given metric.

#To fetch the features for each model of X variables to optimize Kappa
variables_kappa <- results_kappa$variables
#For the following, the number in the index filtering represents the number of features you want
#We use 26 here to have significant feature reduction while retaining good Kappa scores
variables_kappa_numvar <- variables_kappa[variables_kappa$Variables==26,]
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
rose_final_x_kappa$class <- rose_final_x$class


#To fetch the features for each model of X variables to optimize Accuracy
variables_acc <- results_acc$variables
#For the following, the number in the index filtering represents the number of features you want
#We use 29 here to have significant feature reduction while retaining good Accuracy scores
variables_acc_numvar <- variables_acc[variables_acc$Variables==29,]
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
rose_final_x_acc$class <- rose_final_x$class

######################################################
# KNN - USING SMOTE BALANCED DATA, ACCURACY FEATURES #
######################################################
knn_model_acc <- train(class~., data=rose_final_x_acc, method = "knn", tuneLength = 20, 
                       trControl = ctrl, preProc = c("center", "scale"))

#Prediction & Confusion Matrix on Training Data
pred_acc <- predict(knn_model_acc, newdata = training_x_acc)
confusionMatrix(pred_acc, training_y)

#Prediction & Confusion Matrix on Validation Data
pred_acc_valid <- predict(knn_model_acc, newdata = validation_x_acc)
confusionMatrix(pred_acc_valid, validation_y)

#################################################################
# KNN - FINAL MODEL - USING SMOTE BALANCED DATA, KAPPA FEATURES #
#################################################################
knn_model_kappa <- train(class~., data=rose_final_x_kappa, method = "knn", tuneLength = 20, 
                         trControl = ctrl, preProc = c("center", "scale"))

#Prediction & Confusion Matrix on Training Data
pred_kappa <- predict(knn_model_kappa, newdata = training_x_kappa)
confusionMatrix(pred_kappa, training_y)

#Prediction & Confusion Matrix on Validation Data
pred_kappa_valid <- predict(knn_model_kappa, newdata = validation_x_kappa)
confusionMatrix(pred_kappa_valid, validation_y)