# Exercise Quality Prediction Using Machine Learning

## Project Overview
This repository contains an analysis of weight lifting exercise quality using data from accelerometers placed on the belt, forearm, arm, and dumbbell of 6 participants. The goal is to predict the manner in which participants performed barbell lifts (correctly or incorrectly in 5 different ways).

## Data Source
The data for this project comes from the [Weight Lifting Exercise Dataset](http://web.archive.org/web/20161224072740/http://groupware.les.inf.puc-rio.br/har). Six participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions:

- Class A: exactly according to the specification
- Class B: throwing the elbows to the front
- Class C: lifting the dumbbell only halfway
- Class D: lowering the dumbbell only halfway
- Class E: throwing the hips to the front

## Project Structure
- `exercise_quality_prediction.Rmd`: R Markdown file containing all code and analysis
- `exercise_quality_prediction.html`: Compiled HTML output of the analysis
- `pml-training.csv`: Training dataset
- `pml-testing.csv`: Testing dataset with 20 cases for prediction

## Methods Used
- **Preprocessing**: Handling missing values and removing irrelevant features
- **Machine Learning**: Random Forest algorithm with 5-fold cross-validation
- **Model Evaluation**: Confusion matrix and out-of-sample error estimation
- **Feature Importance**: Analysis of the most predictive variables

## Results
The Random Forest model achieved 99.49% accuracy on the validation set, with an estimated out-of-sample error rate of only 0.51%. The most important predictors were orientation measurements, with roll_belt being the most significant predictor, followed by pitch_forearm and yaw_belt.

## Requirements
- R (version >= 4.0.0 recommended)
- Required R packages:
  - caret
  - randomForest
  - dplyr
  - ggplot2
  - corrplot

## Usage
1. Clone this repository
2. Open the `.Rmd` file in RStudio
3. Install required packages if needed:
   ```R
   install.packages(c("caret", "randomForest", "dplyr", "ggplot2", "corrplot"))
   ```
4. Run all code chunks to reproduce the analysis
5. The predictions for the 20 test cases will be generated in the final section

## Citation
If you use this analysis or the dataset, please cite the original authors:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13). Stuttgart, Germany: ACM SIGCHI, 2013.
