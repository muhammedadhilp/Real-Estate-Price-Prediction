# Real Estate Price Prediction

This project demonstrates a real estate price prediction model using a linear regression approach. The dataset is preprocessed, and features are selected and encoded for optimal model performance. We utilize techniques like Mutual Information Score for feature selection, Z-score for outlier detection, and cross-validation to evaluate the model.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Selection](#feature-selection)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Installation

To run the code, you need to have Python installed with the following libraries:

```
pip install numpy pandas matplotlib seaborn scipy scikit-learn category_encoders
```
# Project Structure
data.csv: The dataset containing real estate listings with various features like square footage, number of bedrooms, and price.
housepriceprediction.ipynb: The main Python script that contains all the code for data loading, preprocessing, model training, and evaluation.

## Dataset
The dataset contains various features of real estate properties, including:

street, statezip, city: Categorical data representing the location.
sqft_living, sqft_above, bathrooms, yr_built, sqft_lot, bedrooms: Numerical data representing property features.
price: The target variable representing the property price.
### Exploratory Data Analysis (EDA)
The dataset is loaded and basic information is displayed using df.info(). We analyze the distribution of the target variable price using a Kernel Density Estimate (KDE) plot. The data appears left-skewed, indicating potential outliers.
```
sns.kdeplot(df['price'].apply(np.log1p), fill=True)
plt.show()
```

# Feature Selection
## Mutual Information Score
We calculate the Mutual Information Score to identify the most important features related to the target variable price.
```
def make_mi_score(x, y):
    # Processing categorical variables
    for colname in x.select_dtypes(['object', 'category']):
        x[colname], _ = x[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in x.dtypes]
    mi_score = mutual_info_regression(x, y, discrete_features=discrete_features, random_state=42)
    mi_score = pd.Series(mi_score, name='Mutual Information Score', index=x.columns)
    return mi_score.sort_values(ascending=False)

mi_score = make_mi_score(X, Y)
```

## Visualization
The Mutual Information Scores are visualized to easily interpret feature importance.
```
def plot_mi_score(score):
    score = score.sort_values(ascending=True)
    width = np.arange(len(score))
    ticks = list(score.index)
    plt.barh(width, score)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Score")
    plt.show()

plot_mi_score(mi_score)
```
# Data Preprocessing
## Outlier Detection
Outliers are detected using Z-scores. A threshold of 3 is used to identify and remove significant outliers
```
z = np.abs(stats.zscore(df[['sqft_living', 'sqft_above', 'bathrooms', 'yr_built', 'sqft_lot', 'bedrooms']]))
df = df[(z < 3).all(axis=1)]
```
## Target Encoding
Categorical variables (street, statezip, city) are encoded using MEstimateEncoder, which handles categorical variables better than traditional methods like One-Hot Encoding.
```
encoder = MEstimateEncoder(cols=['street', 'statezip', 'city'], m=0.5)
X = encoder.fit_transform(X, Y)
```
# Model Training and Evaluation
The dataset is split into training and testing sets using an 80-20 split. We train a linear regression model and evaluate it using cross-validation and metrics like Mean Squared Error (MSE) and R-squared (R²).
```
model = LinearRegression()
model.fit(x_train, y_train)

 Cross-validation
cvs = cross_val_score(model, x_train, y_train, cv=10, n_jobs=-1)
print('Accuracy: {:.2f} %'.format(cvs.mean() * 100))

# Model evaluation
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE:', mse)
print('R²:', r2)
```
# Results
MSE: Measures the average of the squares of the errors between predicted and actual values.
R² Score: Indicates how well the independent variables explain the variability of the dependent variable (price).
These metrics suggest the model's performance and its accuracy in predicting real estate prices.

# Conclusion
This project successfully demonstrates the use of linear regression in predicting real estate prices. The process of feature selection, data preprocessing, and model evaluation provides insights into handling real-world datasets for predictive modeling.

This `README.md` file should give anyone a comprehensive understanding of your project, including how to install dependencies, understand the structure, and run the code.
