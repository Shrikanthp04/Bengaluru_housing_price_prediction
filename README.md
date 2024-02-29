# Bengaluru House Price Prediction
This repository contains code for predicting house prices in Bengaluru using Machine Learning models. The dataset used for training and testing the models is the Bengaluru House Price dataset, which includes various features such as size, location, total square feet area, number of bathrooms, and other relevant information.

# Dataset
The dataset used for this project is sourced from Bengaluru House Data. It contains information about houses in Bengaluru and their corresponding attributes. Initially, the dataset is loaded and explored to understand its structure and contents.

* Area_type - Description of the area
* Availability - When it can be possessed or when it is ready
* Location - Where it is located in Bengaluru
* Size - BHK or Bedrooms
* Society - To which society it belongs
* Total_sqft - Size of the property in sq.ft
* Bath - No. of Bathrooms
* Balcony - No. of the Balcony
* Price - Value of the property in lakhs (Indian Rupee - ₹)

# Exploratory Data Analysis (EDA)
Exploratory data analysis is performed to gain insights into the dataset. Various visualizations such as heatmap, boxplots, and distribution plots are utilized to understand the relationships between different features and the target variable (house price).

# Data Preprocessing
1. Dropping Unwanted Columns: Columns like 'availability' and 'society' which are not significant for our analysis are dropped.
2. Handling Missing Values: Missing values in the 'location' column are filled with the mode value. Numeric missing values are imputed using the median.
3. Feature Engineering: New features such as 'BHK' (Bedrooms, Hall, Kitchen) are derived from existing features.
4. Categorical Columns Encoding :-
* Counting Location Occurrences: We determined the frequency of each location in the dataset, prioritizing those with occurrences less than or equal to 10.
* Identifying Locations with Few Occurrences: Locations meeting the criteria of having 10 or fewer occurrences were grouped under the label 'other' to address sparsity and reduce dimensionality.
* One-Hot Encoding: Utilizing one-hot encoding, we converted categorical location data into binary vectors, representing each category as a binary feature (dummy variable).
5. Outlier Handling: Outliers in numerical features are identified and removed using a z-score threshold.

# Model Building
Several machine learning regression models are trained and evaluated for predicting house prices. The following models are explored:

1. Linear Regression
2. Ridge Regression
3. ElasticNet
4. Lasso Regression
5. Decision Tree Regressor
6. Random Forest Regressor
7. AdaBoost Regressor
8. XGB Regressor

### Each model is trained on the training dataset and evaluated using metrics such as Mean Absolute Error (MAE) and R-squared score.

Linear Regression model
**************************************** 
* Model : Linear Regression metrics
* MAE : 25.44
* r2_score : 65.93% 
**************************************** 

Ridge model
**************************************** 
* Model : Ridge metrics
* MAE : 25.33
* r2_score : 65.99% 
**************************************** 

ElasticNet model
**************************************** 
* Model : ElasticNet metrics
* MAE : 27.59
* r2_score : 58.82% 
**************************************** 

Lasso model
**************************************** 
* Model : Lasso metrics
* MAE : 27.39
* r2_score : 59.61% 
**************************************** 

Decision Tree Regressor model
**************************************** 
* Model : Decision Tree Regressor metrics
* MAE : 25.47
* r2_score : 55.60% 
**************************************** 

Random Forest Regressor model
**************************************** 
* Model : Random Forest Regressor metrics
* MAE : 23.44
* r2_score : 64.43% 
**************************************** 

AdaBoost Regressor  model
**************************************** 
* Model : AdaBoost Regressor  metrics
* MAE : 32.21
* r2_score : 54.82% 
**************************************** 

XGBRegressor model
**************************************** 
* Model : XGBRegressor metrics
* MAE : 21.67
* r2_score : 70.84% 
**************************************** 


# Hyperparameter Tuning
Hyperparameter tuning is performed on the XGB Regressor model using GridSearchCV to find the optimal combination of hyperparameters that maximize the model's performance. The best model obtained from hyperparameter tuning is further evaluated on the test dataset.

# Conclusion
After thorough experimentation and optimization, the XGBoost Regressor model, fine-tuned with optimized hyperparameters, has demonstrated superior performance in predicting house prices in Bengaluru, boasting an impressive R-squared score of 71.14%. This model serves as a reliable tool for making precise predictions of house prices based on the provided features. With a mean absolute error of 21.65 units, stakeholders can trust its accuracy within a reasonable margin, making informed decisions confidently.