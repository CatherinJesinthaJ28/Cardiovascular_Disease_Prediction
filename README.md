# Cardiovascular Disease Prediction

## Machine Learning Project

A machine learning project that analyzes cardiovascular health data and
predicts the presence of cardiovascular disease using classification
algorithms.

## Project Overview

This project uses patient health information to explore patterns related
to cardiovascular disease and build machine learning classification
models.

### Workflow

1.  Data analysis
2.  Data preprocessing and cleaning
3.  Exploratory data visualization
4.  Feature selection
5.  Machine learning model training
6.  Model comparison
7.  Decision Tree tuning
8.  Confusion matrix evaluation
9.  Saving the final trained model

## Dataset

The dataset contains patient health-related attributes such as:

-   Age
-   Gender
-   Height
-   Weight
-   Systolic blood pressure (`ap_hi`)
-   Diastolic blood pressure (`ap_lo`)
-   Cholesterol level
-   Glucose level
-   Smoking status
-   Alcohol consumption
-   Physical activity
-   Cardiovascular disease (`cardio`)

The cleaned dataset contains **68,672 records and 12 columns** after
blood-pressure cleaning.

## Data Preprocessing

The preprocessing stage includes:

-   Checking for duplicate rows
-   Detecting suspicious blood-pressure values
-   Removing invalid blood-pressure records
-   Converting age from days to years
-   Preparing the cleaned dataset for machine learning

### Data Cleaning Result

-   Original rows: **70,000**
-   Duplicate rows: **0**
-   Rows after blood-pressure cleaning: **68,672**
-   Rows removed: **1,328**

## Exploratory Data Analysis

The project includes visualizations for:

-   Cardiovascular disease distribution
-   Age distribution
-   Cardiovascular disease by age
-   Cardiovascular disease by gender
-   Cardiovascular disease by cholesterol level
-   Cardiovascular disease by glucose level
-   Systolic blood pressure by cardiovascular disease
-   Diastolic blood pressure by cardiovascular disease
-   Weight by cardiovascular disease
-   BMI by cardiovascular disease
-   Cardiovascular disease by physical activity
-   Cardiovascular disease by smoking status
-   Cardiovascular disease by alcohol consumption
-   Correlation heatmap

All generated plots are available in `Output/Plots/`.

## Machine Learning Models

Three classification algorithms were compared:

-   Logistic Regression
-   Decision Tree
-   Random Forest

### Model Comparison

  Model                   Accuracy
  --------------------- ----------
  Logistic Regression       72.28%
  Decision Tree             72.70%
  Random Forest             70.64%

Based on the tested models, the **Decision Tree** achieved the highest
accuracy in the initial model comparison.

## Decision Tree Tuning

Different maximum-depth values were tested:

    Max Depth   Accuracy
  ----------- ----------
            3     72.25%
            5     72.78%
            7     72.67%
           10     72.29%
           15     70.23%
           20     67.27%

Among the tested depth values, **Max Depth = 5** produced the highest
accuracy of **72.78%**.

## Final Model Evaluation

The final Decision Tree evaluation produced the following confusion
matrix:

``` text
[[5746 1194]
 [2545 4250]]
```

The final evaluation shown in the project output is approximately **72%
accuracy**.

The trained model is saved as:

`Output/final_decision_tree_model.pkl`

## Project Structure

``` text
CARDIOVASCULAR-DISEASE-PREDICTION/
│
├── Dataset/
│   └── cardio_train.csv
│
├── Python Code/
│   ├── 01_data_analysis.py
│   ├── 02_preprocessing.py
│   ├── 03_visualization.py
│   └── 04_model_training.py
│
├── Output/
│   ├── Plots/
│   ├── cardio_cleaned.csv
│   ├── final_confusion_matrix.png
│   └── final_decision_tree_model.pkl
│
├── Report/
│   └── Cardiovascular_Disease_Final_Report.docx
│
├── PPT/
│   └── Cardiovascular_Disease_Presentation.pptx
│
├── README.md
└── requirements.txt
```

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn

## How to Run

### 1. Clone the repository

``` bash
git clone <YOUR-GITHUB-REPOSITORY-LINK>
```

### 2. Open the project folder

``` bash
cd Cardiovascular_Disease_Prediction
```

### 3. Install the required libraries

``` bash
pip install -r requirements.txt
```

### 4. Run data analysis

``` bash
python "Python Code/01_data_analysis.py"
```

### 5. Run preprocessing

``` bash
python "Python Code/02_preprocessing.py"
```

### 6. Generate visualizations

``` bash
python "Python Code/03_visualization.py"
```

### 7. Train and evaluate the models

``` bash
python "Python Code/04_model_training.py"
```

## Project Outputs

The repository contains:

-   Cleaned dataset
-   Exploratory data analysis plots
-   Correlation heatmap
-   Model comparison results
-   Final confusion matrix
-   Trained Decision Tree model
-   Project report
-   Project presentation

## Author

**Catherin Jesintha J**

B.Sc. Computer Science\
St. Anne's Arts and Science College

## Academic Project

Machine Learning Project
