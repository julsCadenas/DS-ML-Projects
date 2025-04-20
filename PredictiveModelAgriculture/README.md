# Investigation Netflix Movies
- A short activity from the *[Machine Learning Engineer Career Track](https://app.datacamp.com/learn/career-tracks/machine-learning-engineer)* from DataCamp.

### **Instructions:**
1. Read the data into a pandas DataFrame and perform exploratory data analysis
- Read in the "soil_measures.csv" file as pandas DataFrame.
- Read in a csv file
- Check for missing values
- Check for crop types

2. Split the data
- Create training and test sets using all features.
- Features and target variables
- Use train_test_split()

3. Evaluate feature performance
- Predict the crop using each feature individually. You should build a model for each feature. That means you will build four models.
- Create a dictionary to store each features predictive performance
- Loop through the features
- Training a multi-class classifier algorithm
- Predicting target values using the test set
- Evaluating the performance of each feature

4. Create the best_predictive_feature variable
- Store the feature name as a key and the respective model's evaluation score as the value.
- Saving the information