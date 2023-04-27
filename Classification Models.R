# ---------------------------------------------------------------------------------------------------------------- #
# Description: Using the classification models below we will predict the type of cancer i.e. “Benign” or “Malignant”
# Developed by Pavlos Poli
# ---------------------------------------------------------------------------------------------------------------- #

# Load required packages
library(tidyverse) # Cleaning data
library(skimr)
library(janitor) # Simple Tools for Examining and Cleaning Dirty Data
library(caret) # Classification and Regression Training
library(GGally) # Extension to ggplot2
library(gridExtra) # Miscellaneous Functions for "Grid" Graphics
library(corrplot) # Plotting correlations
library(randomForest) # ML method for forecasting
library(rpart.plot) # Extension to rpart
library(MASS) # Discriminant Analysis
library(bootStepAIC) # Choose predictor variables
library(e1071) # Misc Functions of the Department of Statistics
library(pROC) # Display and Analyze ROC Curves
library(rpart) # Recursive Partitioning and Regression Trees
library(data.table) # Extension of `data.frame`
library(ROCR) # Visualizing the Performance of Scoring Classifiers
library(rms) # Regression Modeling Strategies
library(ggpubr) # Extension to ggplot2

# Set the environment
set.seed(1234)
rm(list = ls()) # Clear environment
setwd("E:/backup17092018/Myappdir/Myprojects/Data Science/R Scripts/")

# Load dataset
cancer_classify <- read_csv("wisc_bc_data-KNN.csv", col_names = TRUE)

# Explore data structure
skim(cancer_classify)

head(cancer_classify)
tail(cancer_classify)
view(cancer_classify)

# Cleaning operations
cancer_classify <- clean_names(cancer_classify) # Clean field names
cancer_classify <- remove_empty(cancer_classify, c("rows", "cols"), quiet = FALSE) # Remove empty rows
cancer_classify %>%
  mutate_if(is.character, trimws) # Trim string values

# Remove unwanted variable(s) from the dataset
cancer_classify$id <- NULL
view(cancer_classify)

# Split dataset into training set and test set
splitSample <- sample(1:2, size = nrow(cancer_classify), prob = c(0.7, 0.3), 
                      replace = TRUE)

# Training set
train_set <- cancer_classify[splitSample == 1, ]

# Second training set for cross validation
intrain <- sample(1:2, size = nrow(train_set), prob = c(0.7, 0.3), replace = TRUE)
trainset <- train_set[intrain == 1, ]

# Validation set
validset <- train_set[intrain == 2, ]

# Test set
testset <- cancer_classify[splitSample == 2, ]

# I am using K- fold cross validation, with number of folds (k) set at 10
tcontrol <- trainControl(method = "cv", number = 10)
set.seed(1234)

# Fitting training set to classification models using k=10 cross validation

# KNN
modelKNN <- train(diagnosis ~ ., data = trainset, method = "knn", preProcess = c("center", 
                                                                                 "scale"), trControl = tcontrol)
# Naive Bayes
modelNB <- train(diagnosis ~ ., data = trainset, method = "nb", trControl = tcontrol)

# Random Forest
modelRF <- train(diagnosis ~ ., data = trainset, method = "rf", ntree = 100, 
                 importance = TRUE, trControl = tcontrol)

# Logistic Regression
modelLG <- train(diagnosis ~ ., data = trainset, method = "glm", family = binomial, 
                 trControl = tcontrol)

# Make use of the train models and make predictions on validation set

# KNN
pKNN <- predict(modelKNN, validset)

# Naive Bayes
pNB <- predict(modelNB, validset)

# Random Forest
pRF <- predict(modelRF, validset)

# Logistic Regression
pLG <- predict(modelLG, validset)

# Create a confusion matrix for each model

# KNN
cmKNN <- confusionMatrix(as.factor(validset$diagnosis), pKNN)

# Naive Bayes
cmNB <- confusionMatrix(as.factor(validset$diagnosis), pNB)

# Random Forest
cmRF <- confusionMatrix(as.factor(validset$diagnosis), pRF)

# Logistic Regression
cmLG <- confusionMatrix(as.factor(validset$diagnosis), pLG)

# Create a table of all the values

# Create a vector containing names of models
ModelType <- c("K nearest neighbor", "Naive Bayes", "Random forest", "Logistic regression")

# Training classification accuracy
TrainAccuracy <- c(max(modelKNN$results$Accuracy), max(modelNB$results$Accuracy), 
                   max(modelRF$results$Accuracy), max(modelLG$results$Accuracy))

# Training misclassification error
Train_missclass_Error <- 1 - TrainAccuracy

# validation classification accuracy
ValidationAccuracy <- c(cmKNN$overall[1], cmNB$overall[1], cmRF$overall[1], 
                        cmLG$overall[1])

# Validation misclassification error or out-of-sample-error
Validation_missclass_Error <- 1 - ValidationAccuracy

# Create a data frame with the above metrics
metrics <- data.frame(ModelType, TrainAccuracy, Train_missclass_Error, ValidationAccuracy, 
                      Validation_missclass_Error)

# Print table the table using kable() from knitr package
knitr::kable(metrics, digits = 5)

# Use Random Forest to predict values from test set
pTestingRF <- predict(modelRF, testset)
pTestingRF

# Use KNN to predict values from test set
pTestingKNN <- predict(modelKNN, testset)
pTestingKNN

# Write predicted values form KNN model to dataset
cancer_classify$diagnosis_prediction <- predict(modelKNN, cancer_classify)
view(cancer_classify)

# Write the modeled dataset to .CSV
write_csv(cancer_classify, "wisc_bc_data-KNN-with-Predictions.csv")
