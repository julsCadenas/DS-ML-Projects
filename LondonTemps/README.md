# Predicting Temperature in London
- A short activity from the *[Machine Learning Engineer Career Track](https://app.datacamp.com/learn/career-tracks/machine-learning-engineer)* from DataCamp.

### **Instructions:**
1. Loading the data
- Load the data from `london_weather.csv` to understand its contents.
- Reading in the data.
- Determine the column names, data types, number of null values.

2. Data cleaning
- Convert columns to the correct data type to enable exploratory data analysis.
- Working with the date column.
- Extracting more date information.

3. Exploratory Data Analysis
- Explore the data by visualizing some of the features.
- Visuzalizing temperature.

4. Feature Selection
- Choose appropriate features to predict the mean temperature.
- Filter features.

5. Preprocess data
- Use an imputer to account for missing values, then normalize the features using the scaler. Make sure to split the data into train and test samples at the right moment. These preprocessing steps should go into a pipeline.
- Imputing and normalizing data.

6. Machine learning training and evaluation
- Try out regression models such as linear regression, decision tree, and random forest regressors, with various hyperparameters, to find the best performing model. Log all of your models and metrics using MLflow.
- Building a for loop to try different hyperparameters.
- Logging and evaluating.

7. Searching your logged results
- Create a variable called `experiment_results` containing all logged data relating to your MLflow runs.
- Searching runs.