#=-=-=-=
library(tree)
library(party)
library(randomForest)
library(caret)
library(ROSE)
library(Matrix)
library(xgboost)
library(dplyr)

#Converting to R DataFrame Objects
training_df <- as.data.frame(training_data)
test_df <- as.data.frame(test_data)

#Splitting the training data into Training (75%) and Validation (25%)
set.seed(120)
ntrain_full<- round(nrow(training_data)*0.75)  # 75% for training set
index_full <- sample(nrow(training_data),ntrain_full)

training_x_5<- training_data[index_full,]
training_x_5$BF_AreaShape_Area <- as.numeric(training_x_5$BF_AreaShape_Area)
training_x_5$class <- factor(training_x_5$class)

validation_x_5<- training_data[-index_full,]
validation_x_5$BF_AreaShape_Area <- as.numeric(validation_x_5$BF_AreaShape_Area)
validation_x_5$class <- factor(validation_x_5$class)

#########
# SMOTE #
#########
#ROSE package is used here to have full control over balancing.
#ROSE only allows binary classification, so for each step this classification is 'reset'.

#STEP 1 - Oversample Anaphase (10 --> 6600)
training_x_5$binary <- 0
training_x_5$binary[training_x_5$class == "Anaphase"] <- 1
training_x_5$binary <- as.factor(training_x_5$binary)

rose_under_5 <- ovun.sample(binary ~ ., data = training_x_5, 
                          method = "over",N = 28360)$data

#STEP 2 - Undersample Prophase (401 --> 70)
rose_under_5$binary <- 0
rose_under_5$binary[rose_under_5$class == "Prophase"] <- 1
rose_part_5 <- rose_under_5[rose_under_5$class != "Anaphase",]
rose_part_5_2 <- rose_part_5[rose_part_5$class != "Telophase",]
rose_part_5_3 <- rose_part_5_2[rose_part_5_2$class != "Interphase",]
rose_part_5_3$binary <- as.factor(rose_part_5_3$binary)

rose_under_5_step2 <- ovun.sample(binary ~ ., data = rose_part_5_3, 
                                method = "under",N = 120)$data

#STEP 3 - Oversample Metaphase (50  --> 250)
rose_under_5_step2$binary <- 0
rose_under_5_step2$binary[rose_under_5_step2$class == "Metaphase"] <- 1
rose_part2_5 <- rose_under_5_step2[rose_under_5_step2$class != "Anaphase",]
rose_part2_5_2 <- rose_part2_5[rose_part2_5$class != "Telophase",]
rose_part2_5_3 <- rose_part2_5_2[rose_part2_5_2$class != "Interphase",]

rose_under_5_step3 <- ovun.sample(binary ~ ., data = rose_part2_5_3, 
                                  method = "over",N = 320)$data

# STEP 4 - Undersample Interphase (21292 --> 200)
rose_under_5_step2$binary <- 0
rose_under_5_step2$binary[rose_under_5_step2$class == "Interphase"] <- 1
rose_part3_5 <- rose_under_5[rose_under_5$class != "Anaphase",]
rose_part3_5_2 <- rose_part3_5[rose_part3_5$class != "Telophase",]
rose_part3_5_3 <- rose_part3_5_2[rose_part3_5_2$class != "Prophase",]
rose_part3_5_3$binary <- as.factor(rose_part3_5_3$binary)

rose_under_5_step4 <- ovun.sample(class ~ ., data = rose_part3_5_3, 
                                  method = "under",N = 250)$data

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
rm(rose_part_5, rose_part_5_2, rose_part_5_3, rose_part1_5,rose_part2_5, rose_part2_5_2, rose_part2_5_3,
   rose_part3_5, rose_part3_5_2, rose_part3_5_3, rose_partA_5, rose_partA2_5, rose_partB_5, rose_partC_5,
   rose_partD_5, rose_under_5, rose_under_5_step2, rose_under_5_step3,rose_under_5_step4)

#Splitting predictor features to X and classification label to Y
rose_final_y <- rose_final_5$class
rose_final_x <- rose_final_5

#Predictors X should not have 'class' nor 'binary' (latter created for SMOTE)
rose_final_x$class <- NULL
rose_final_x$binary <- NULL

#Neither should the vector 'binary' be used again for modeling purposes
rose_final_5$binary <- NULL

################################
# NON-CORRELATED FEATURES ONLY #
################################
corr_matrix <- cor(rose_final_x)
highlyCorrelated <- findCorrelation(corr_matrix, cutoff = 0.75)
print(highlyCorrelated)
rose_final_x_noncorr <- rose_final_x[,-highlyCorrelated]

###########
# XGBOOST #
##########
#Creating compatible Matrix objects for XGBoost R package

#Training data matrix, using the SMOTE data and non-correlated features
trainm <- sparse.model.matrix(rose_final_y ~., data = rose_final_x_noncorr)
train_label <- rose_final_y
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)

#Validation data matrix, using original imbalanced data and non-correlated features
validation_x_5_noncorr <- validation_x_5[,-highlyCorrelated]
validation_y <- validation_x_5$class
testm <- sparse.model.matrix(validation_y~., data = validation_x_5_noncorr)
test_label <- validation_y
test_matrix <- xgb.DMatrix(data = as.matrix(testm), label = test_label)

#Parameters (you can use multi:softprob or multi:softmax for multi-classification problems)
nc <- length(unique(train_label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = 6) #in R, number of classes + 1 must be used for compatibility
watchlist <- list(train = train_matrix, test = test_matrix)

#Training the first XGBoost Model
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nround = 511,
                       watchlist = watchlist,
                       eta = 0.03,
                       max.depth = 15,
                       gamma = 0,
                       subsample = 1,
                       seed = 120)

# Training & test error plot
e <- data.frame(bst_model$evaluation_log)
plot(e$iter, e$train_mlogloss, col = 'blue')
lines(e$iter, e$test_mlogloss, col = 'red')
#Interpretation of the plot - Error Rates for validation data are higher, indicates overfitting of model

#Find the iteration number with lowest error rate on validation data
min(e$test_mlogloss)
e[e$test_mlogloss == min(e$test_mlogloss),]

#Feature importance
imp <- xgb.importance(colnames(train_matrix), model = bst_model)
print(imp)
xgb.plot.importance(imp)
#Only a handful of features seem to have great contribution to the model


#Prediction & Confusion Matrix - test data
p <- predict(bst_model, newdata = test_matrix)
pred <- matrix(p, nrow = 6, ncol = length(p)/6) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label, max_prob = max.col(., "last")-1)

table(Prediction = pred$max_prob, Actual = pred$label)

#FEATURE SELECTION FOR NEXT ITERATION
#Fetch the variable names and their 'gain' importance metric in XGBoost package
xgb_features <- imp[,1:2]
rownames(xgb_features) <- xgb_features$Feature
xgb_features <- xgb_features[xgb_features$Gain > 0.005]

#Modifying Training Data with lesser, more important features
rose_xgb_features <- rose_final_x_noncorr[,xgb_features$Feature]

#Training data matrix, using the SMOTE data and XGBoost Importance Based Features
trainm_xgb <- sparse.model.matrix(rose_final_y ~., data = rose_xgb_features)
head(trainm_xgb)
train_label_xgb <- rose_final_y
train_matrix_xgb <- xgb.DMatrix(data = as.matrix(trainm_xgb), label = train_label_xgb)

#Similar with Validation data - original class imbalance, XGBoost Importance Based Features
validation_xgb_features <- validation_x_5[,xgb_features$Feature]
testm_xgb <- sparse.model.matrix(validation_y ~., data = validation_xgb_features)
test_label_xgb <- validation_y
test_matrix_xgb <- xgb.DMatrix(data = as.matrix(testm_xgb), label = test_label_xgb)

#Copy of parameters for a new XGBoost Model
nc_xgb <- length(unique(train_label_xgb))
xgb_params_xgb <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = 6)
watchlist_xgb <- list(train = train_matrix_xgb, test = test_matrix_xgb)

#But before we train another XGBoost model with only these features, let us perform cross-validation.
#This is done using the training data matrix of all non-correlated features.
#Purpose is to seek for a right number of iterations to minimize the error.

#USING XGB.CV - FINAL TUNING OF PARAMETERS ALLOWED
bst_model_cv <- xgb.cv(params = xgb_params,
                       data = train_matrix,
                       nround = 1000, #tuned using CV
                       nfold = 5,
                       watchlist = watchlist,
                       prediction = TRUE,
                       verbose = TRUE,
                       early_stopping_rounds = 10,
                       max.depth = 15, #tuned using CV
                       eta = 0.03,
                       gamma = 0,
                       subsample = 1,
                       seed = 120)

#cv error (usually too optimistic)
min(bst_model_cv$evaluation_log[,4])
#Best found was 0.0184172, max.depth = 15, nround = 511, eta = 0.03, gamma = 0

e_xgbcv <- data.frame(bst_model_cv$evaluation_log)
plot(e_xgbcv$iter, e_xgbcv$train_mlogloss_mean, col = 'blue')
lines(e_xgbcv$iter, e_xgbcv$test_mlogloss_mean, col = 'red')
#overfitting has been reduced

#Final Model - Using XGBoost Importance Based Variables
#We use the same number of rounds as derived in the cross-validated XBGoost Model
#Other parameter settings are copied over.
bst_model_xgb <- xgb.train(params = xgb_params_xgb,
                       data = train_matrix_xgb,
                       nround = 511,
                       watchlist = watchlist_xgb,
                       eta = 0.03,
                       max.depth = 15,
                       gamma = 0,
                       subsample = 1,
                       seed = 120)

#Training & test error plot
e_xgb <- data.frame(bst_model_xgb$evaluation_log)
plot(e_xgb$iter, e_xgb$train_mlogloss, col = 'blue')
lines(e_xgb$iter, e_xgb$test_mlogloss, col = 'red')
#Overfitting still takes place

#Feature importance
imp_xgb <- xgb.importance(colnames(train_matrix_xgb), model = bst_model_xgb)
print(imp_xgb)
xgb.plot.importance(imp_xgb)
#Feature Granularity_1_DF_image with largest 'gain' metric value, but more features are needed to have
#decent classification performance for all classes

#Prediction & Confusion Matrix - Validation Data
p_xgb <- predict(bst_model_xgb, newdata = test_matrix_xgb)
pred_xgb <- matrix(p_xgb, nrow = 6, ncol = length(p_xgb)/6) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label_xgb, max_prob = max.col(., "last")-1)

#The final Confusion Matrix used for Model Performance Analysis
table(Prediction = pred_xgb$max_prob, Actual = pred_xgb$label)
