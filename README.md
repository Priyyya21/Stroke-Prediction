##  Stroke Risk Prediction using Machine Learning

This project predicts the likelihood of a stroke based on medical and lifestyle attributes using a trained Random Forest Classifier, which achieved the best performance (98.82%) among all tested models.

####  Live Web App:
https://stroke-prediction-priya.onrender.com

 ##  Project Overview

Stroke is a major global health concern and early prediction can help prevent critical outcomes.
In this work, machine learning is applied to analyze key risk factors like:
- Age
- Hypertension
- Heart Disease
- Smoking Status
- BMI
- Glucose Level
- Work & Residence type
- Marital status
- Gender

The model outputs the probability (%) of having a stroke, along with a risk label (Low, Moderate, High).

##  Models Used & Accuracy Comparison
##### Model Accuracy
Decision Tree Classifier: 96%;
Logistic Regression: 76%;
XGBoost Classifier: 97%;
Random Forest Classifier: 98.82% (Best)

Based on performance, Random Forest was selected for deployment.

##  Tech Stack
##### Component:	Technology
Frontend:        	      HTML, CSS; 
Backend:        	        Flask (Python);
ML Model:	                RandomForestClassifier (scikit-learn);
Deployment:               	Render;
Data Handling:	            pandas, numpy;
Model Serialization:      	joblib;

## Dataset
The dataset used is from the Stroke Prediction Dataset
source: Kaggle / Open Healthcare Dataset

Includes 10 key input features after preprocessing.


## Clone the repo

git clone: https://github.com/PriyanshiSangwan/stroke-detection


## Features

- Real-time Stroke Prediction
- Displays Risk Level + Probability
- Responsive design


	
##  Disclaimer

This tool is meant for educational purposes only
It is not medical advice and should not replace consultation with a qualified healthcare professional.

## Developer

Priya Sharma
B.Tech CSE — Machine Learning & Data Science Enthusiast 

 Email: priyaax21@gmail.com
 LinkedIn: https://www.linkedin.com/in/priya-vashishth-1790512b2/


 
