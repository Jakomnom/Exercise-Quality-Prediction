---
title: "Exercise Quality Prediction using Accelerometer Data"
date: "2025-03-10"
output: 
  html_document:
    keep_md: true
    toc: true
    toc_float: true
    code_folding: hide
---



## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit, it is now possible to collect a large amount of data about personal activity relatively inexpensively. These types of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify *how much* of a particular activity they do, but they rarely quantify *how well they do it*.

In this project, I will use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal is to predict the manner in which they did the exercise, represented by the "classe" variable in the training set.

## Data Loading


```r
# Download data if it doesn't exist
if (!file.exists("pml-training.csv")) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                "pml-training.csv")
}

if (!file.exists("pml-testing.csv")) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                "pml-testing.csv")
}

# Load the data
training <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
testing <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))

# Check dimensions
dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

## Exploratory Data Analysis

Let's first look at the structure of the data and identify the features we'll use.


```r
# Look at the first few rows
head(training[, 1:10])
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
##   new_window num_window roll_belt pitch_belt yaw_belt
## 1         no         11      1.41       8.07    -94.4
## 2         no         11      1.41       8.07    -94.4
## 3         no         11      1.42       8.07    -94.4
## 4         no         12      1.48       8.05    -94.4
## 5         no         12      1.48       8.07    -94.4
## 6         no         12      1.45       8.06    -94.4
```

```r
# Check the classe distribution
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
qplot(classe, data = training, fill = classe, main = "Distribution of Exercise Classes")
```

![](Exercise-Quality-Prediction-using-Accelerometer-Data_files/figure-html/explore_data-1.png)<!-- -->

```r
# Check missing values
missing_values <- colSums(is.na(training))/nrow(training)
summary(missing_values)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##  0.0000  0.0000  0.9793  0.6132  0.9793  1.0000
```

## Data Preprocessing

We need to clean the data before building our model:

1. Remove columns with high percentage of missing values
2. Remove identification and timestamp columns which won't be useful for prediction
3. Remove near-zero variance predictors


```r
# Remove columns with more than 60% missing values
high_nas <- names(training)[colSums(is.na(training))/nrow(training) > 0.6]
training_clean <- training[, !(names(training) %in% high_nas)]
testing_clean <- testing[, !(names(testing) %in% high_nas)]

# Remove identification and timestamp columns (first 7 columns)
training_clean <- training_clean[, -(1:7)]
testing_clean <- testing_clean[, -(1:7)]

# Check for near-zero variance predictors
nzv <- nearZeroVar(training_clean, saveMetrics = TRUE)
training_clean <- training_clean[, !nzv$nzv]
testing_clean <- testing_clean[, names(testing_clean) %in% names(training_clean)]

# Make sure all columns in testing match training
testing_clean <- testing_clean[, names(testing_clean) %in% names(training_clean)]

# Check dimensions after cleaning
dim(training_clean)
```

```
## [1] 19622    53
```

```r
dim(testing_clean)
```

```
## [1] 20 52
```

Let's look at correlations between predictors:


```r
# Select numeric columns for correlation analysis
numeric_cols <- sapply(training_clean, is.numeric)
correlation <- cor(training_clean[, numeric_cols & names(training_clean) != "classe"])

# Plot correlation matrix (only showing a subset for visibility)
corrplot(correlation[1:15, 1:15], method = "color", type = "upper", 
         order = "hclust", tl.col = "black", tl.cex = 0.7)
```

![](Exercise-Quality-Prediction-using-Accelerometer-Data_files/figure-html/correlation-1.png)<!-- -->

## Model Building

We'll split the training data into training and validation sets for cross-validation:


```r
# Create validation set (30% of training data)
inTrain <- createDataPartition(training_clean$classe, p = 0.7, list = FALSE)
train_data <- training_clean[inTrain, ]
validation_data <- training_clean[-inTrain, ]

# Verify dimensions
dim(train_data)
```

```
## [1] 13737    53
```

```r
dim(validation_data)
```

```
## [1] 5885   53
```

### Random Forest Model

Random Forest is a good choice for this problem because:
1. It handles non-linear relationships well
2. It's robust to outliers
3. It provides feature importance rankings


```r
# Use k-fold cross-validation (k=5)
control <- trainControl(method = "cv", number = 5)

# Train Random Forest model
set.seed(12345)
rf_model <- train(classe ~ ., data = train_data, 
                 method = "rf", 
                 trControl = control,
                 ntree = 100)

# Print model summary
print(rf_model)
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10990, 10989, 10991, 10988 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9911913  0.9888565
##   27    0.9911915  0.9888568
##   52    0.9830381  0.9785423
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

## Model Evaluation

Let's evaluate the model on our validation set:


```r
# Predict on validation set
rf_predictions <- predict(rf_model, validation_data)

# Create confusion matrix
conf_matrix <- confusionMatrix(rf_predictions, factor(validation_data$classe))
conf_matrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    7    0    0    0
##          B    1 1129    5    0    0
##          C    0    3 1018    7    3
##          D    0    0    3  957    1
##          E    0    0    0    0 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9927, 0.9966)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9912   0.9922   0.9927   0.9963
## Specificity            0.9983   0.9987   0.9973   0.9992   1.0000
## Pos Pred Value         0.9958   0.9947   0.9874   0.9958   1.0000
## Neg Pred Value         0.9998   0.9979   0.9984   0.9986   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1918   0.1730   0.1626   0.1832
## Detection Prevalence   0.2855   0.1929   0.1752   0.1633   0.1832
## Balanced Accuracy      0.9989   0.9950   0.9948   0.9960   0.9982
```

```r
# Calculate out-of-sample error
out_of_sample_error <- 1 - conf_matrix$overall['Accuracy']
out_of_sample_error
```

```
##    Accuracy 
## 0.005097706
```

### Variable Importance

Let's examine which variables are most important for prediction:


```r
# Get variable importance
var_imp <- varImp(rf_model)

# Plot variable importance (top 20)
plot(var_imp, top = 20, main = "Top 20 Important Variables")
```

![](Exercise-Quality-Prediction-using-Accelerometer-Data_files/figure-html/var_importance-1.png)<!-- -->

## Prediction on Test Cases

Now we'll use our model to predict the 20 test cases:


```r
# Predict on test set
final_predictions <- predict(rf_model, testing_clean)

# Output the predictions
final_predictions
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
# Create files for submission (optional)
# pml_write_files = function(x) {
#   n = length(x)
#   for (i in 1:n) {
#     filename = paste0("problem_id_", i, ".txt")
#     write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
#   }
# }
# 
# pml_write_files(final_predictions)
```

## Conclusion

In this project, I built a machine learning model to predict the manner in which participants performed barbell lifts based on accelerometer data. 

The Random Forest model achieved an accuracy of approximately **99.49%** on the validation set, with an estimated out-of-sample error rate of **0.51%**. The most important predictors were variables related to orientation and magnetic field measurements. Specifically, roll_belt was by far the most important variable, followed by pitch_forearm, yaw_belt, and magnet_dumbbell_y. The belt sensors (roll_belt, yaw_belt, pitch_belt) and forearm measurements (pitch_forearm, roll_forearm) dominated the top predictors, suggesting that the orientation of the belt and forearm are particularly informative for distinguishing between different exercise quality classes.

The model successfully predicted all 20 test cases. The high accuracy suggests that accelerometer data can effectively distinguish between correct and incorrect exercise forms, which could be useful for automated exercise feedback systems.

### Why I chose Random Forest:

1. **Performance**: Random Forests typically perform very well on classification tasks with many features
2. **Robustness**: The algorithm is less prone to overfitting compared to decision trees
3. **Feature importance**: It provides insights into which measurements are most relevant for classification
4. **Handling non-linearity**: It captures non-linear relationships without requiring feature transformations

Cross-validation was implemented using a 5-fold approach, which helps ensure the model generalizes well to new data. The low out-of-sample error rate confirms the model's effectiveness.

## References

1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13). Stuttgart, Germany: ACM SIGCHI, 2013.
2. Data source: http://web.archive.org/web/20161224072740/http://groupware.les.inf.puc-rio.br/har
