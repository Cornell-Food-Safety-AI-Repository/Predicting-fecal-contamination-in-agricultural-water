# Environmental-Factors-Water-Quality-Analysis
Welcome to the Environmental Factors Water Quality Analysis repository. This project is dedicated to investigating the impact of various environmental and anthropogenic factors on water quality. Using data extracted from publicly-available databases, this repository contains analyses that explore how elements like campgrounds, culverts, dams, livestock operations, and more can influence the quality of water in different ecosystems.

## Data Overview
Our analysis is grounded in two primary resources:

## Data Dictionary: 
A comprehensive guide explaining the factors considered in our analysis, including their descriptions, data extraction dates, and citations. This dictionary covers a wide range of factors, from natural features like campgrounds and dams to human-related factors such as livestock operations and wastewater discharge sites.

mstdata.csv: A dataset containing measurements and observations relevant to the study. This data serves as the foundation for our analysis, providing quantitative insight into the presence, density, and minimum distance of various factors relative to water bodies.

## Analysis Focus
We utilize a multidisciplinary approach to understand how different factors contribute to water quality. The project looks into:

Presence and Density: Identifying if certain factors are present upstream and calculating their density per 10 km².
Proximity: Determining the flow path distance to the nearest feature of each type.
Water Quality Indicators: Examining parameters like E. coli concentrations, microbial source tracking markers, conductivity, dissolved oxygen, and more.

# Machine Learning Model Predictor

This repository contains a Python script designed to perform predictions using various machine learning models based on user-specified target labels and algorithms. The script supports several regression models and can handle both numerical and categorical data through preprocessing steps like imputation and one-hot encoding.

## Features

- **Data Preprocessing**: Automatic handling of numerical and categorical data including missing value imputation and one-hot encoding.
- **Dynamic Model Selection**: Users can select from multiple regression models to apply on their specified target label.
- **Command Line Interface**: The script is executable from the command line, allowing users to specify the target label and choice of algorithm dynamically.
- **Binary and Continuous Predictions**: Supports both continuous and binary predictions depending on the chosen target label.

## Prerequisites

Before you run this script, make sure you have the following installed:
- Python 3.6 or higher
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn

You can install the necessary libraries using pip:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```

## Usage 
- ** Command Line Execution: Run the script from the command line by specifying the target label and the algorithm you want to use.

``` bash
python ml_model_predictor.py --target_label <target_label> --algorithm <algorithm_name>
```
## Supported Algorithms
- ** RandomForest: Random Forest Regression
- ** LinearRegression: Linear Regression
- ** SVM: Support Vector Machines for regression
- ** GradientBoosting: Gradient Boosting Regressor

## Example
``` bash
python ml_model_predictor.py --target_label remainder__ecoli --algorithm RandomForest
```
## Outputs
- ** The script outputs the model's performance metrics directly to the console, including MSE and R^2 for continuous targets, or accuracy for binary targets.
- ** For binary targets, predictions are automatically converted to binary outcomes based on a threshold.






