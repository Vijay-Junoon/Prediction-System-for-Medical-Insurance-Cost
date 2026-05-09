# Prediction-System-for-Medical-Insurance-Cost


### Purpose
I wanted to learn Multiple Linear Regression and the data preprocessing aspects related to it. So I decided to do this small project.

### Dataset Used
I downloaded a dataset - "Medical Insurance Cost Dataset" from kaggle.

Link : https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset

---

## Pre-processing:

### Libraries
I have imported the following libraries: 
<ul>
  <li>numpy</li>
  <li>pandas</li>
  <li>ColumnTransformer</li>
  <li>OneHotEncoder</li>
  <li>train_test_split</li>
  <li>LinearRegression</li>
  <li>mean_average_error</li>
  <li>mean_squared_error</li>
  <li>r2_score</li>
</ul>

### Handling Missing data
The dataset has no such missing values in any of the columns. So Imputing wasn't necessary.

### Encoding Categorical Data
The dataset had three categorical columns in the feature matrix- Sex, Smoker, Region
Target was numerical.
So I decided to One Hot Encode all the categorical values.
This was done by transforming all columns using ColumnTransformer and by specifying OneHotEncoder()

### Splitting dataset
The ideal split size was found to be 80-20. So I split the dataset into 80% training, and 20% testing data using the train-test_split method in sklearn library under the model_selection module.

---

## Creation of Model and Prediction

A Multiple Linear Regression Model was chosen and it was fit with the training datasets and prediction was done using the testing set.

## Evaluation of Model

The model was evaluated using metrics such as MeanAverageError, MeanSquaredError and R2_Score.
