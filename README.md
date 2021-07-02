# Neural_Network_Charity_Analysis
# Overview
Here we develop and test Neural Network/Machine Learning algorithms against a dataset of charity organizatons with information on their success/failures. After cleaning the data, scaling, binary encoding, etc. we run the data through a Neural Network using Tensorflow, and various Machine Learning Models using Scikit-Learn. We tweak/modify the process numerous times to improve accuracy, in an attempt to create a model good enough to be usable on future data.

# Results

Our target variable in the dataset is the "IS_SUCCESSFUL" column, which is narrowed down to either 1 or 0, and the remaining columns are trained/tested in our models against this, once also numerically encoded; such as organization type, income amount, etc. We also graph a density plot to determine which application/classification types to group together by their count, so they don't create bias/disrupt the accuracy once run through various models:
![image](https://user-images.githubusercontent.com/79726572/124324983-e07c2b80-db51-11eb-9a04-b2f9fac4ab3b.png)
This tells us that the "T" application types with counts less than 500, can be binned into one unique value (simply "other") so they are fairly accounted for, and can be updated on our dataset:
```
replace_application = list(application_counts[application_counts < 500].index)

for app in replace_application:
    application_df.APPLICATION_TYPE = application_df.APPLICATION_TYPE.replace(app,"Other")
    
# Check to make sure binning was successful
application_df.APPLICATION_TYPE.value_counts()

T3       27037
T4        1542
T6        1216
T5        1173
T19       1065
T8         737
T7         725
T10        528
Other      276
Name: APPLICATION_TYPE, dtype: int64
```
This same proccess is also done for the classification column in the dataset.

# Summary
Our initial run through of the TensoFlow model with our data gives us a result of:
```
Loss: 0.5580791234970093, Accuracy: 0.7269970774650574
```
We then exhibit various attempts with scikit models:
```
Logistic Regression Attempt:

log_classifier = LogisticRegression(solver="lbfgs",max_iter=200)
log_classifier.fit(X_train_scaled,y_train)
y_pred = log_classifier.predict(X_test_scaled)
print(f" Logistic regression model accuracy: {accuracy_score(y_test,y_pred):.3f}")
 Logistic regression model accuracy: 0.720
 
Support Vector Machine Attempt:

svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)
print(f" SVM model accuracy: {accuracy_score(y_test,y_pred):.3f}")
 SVM model accuracy: 0.716
 
Random Forest Attempt:

rf_model = RandomForestClassifier(n_estimators=50, random_state=1)
rf_model = rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
print(f" Random forest predictive accuracy: {accuracy_score(y_test,y_pred):.3f}")
 Random forest predictive accuracy: 0.709
```
We then lastly attempt the TensorFlow model with various modifications, such as adding/removing hidden layers, increasing/decreasing number of neurons/epochs, changing activation functions, etc. all to still only achieve an approximate loss of 0.56 and an accuracy of around 0.725

What we can conclude is that further experimenting can be done, even removing columns/additional cleaning to the dataset could be attempted. But what's most likely required is additional data, it's clear that only so much can be done with what we have, even a 75-80 % accuracy isn't enough to be used in a professional setting but at least the initial necessary research has been conducted.
