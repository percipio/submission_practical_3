# Practical Application III: Comparing Classifiers

## What is this?

I am enrolled in an awesome professional certificate program through UC Berkeley. It's an intensive Machine Learning & AI program that I'm learning an incredible amount from.

This is my submission to the practical #3, in Module 17, near the end of the program.

Some of the content in this README is copied directly from the assignment for practical reasons, not wanting to duplicate effort.



**Overview**: In this practical application, your goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.  We will utilize a dataset related to marketing bank products over the telephone.
When 

### Getting Started

Our dataset comes from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing).  The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns.  We will make use of the article accompanying the dataset [here](CRISP-DM-BANK.pdf) for more information on the data and features.

### Problem 1: Understanding the Data

To gain a better understanding of the data, please read the information provided in the UCI link above, and examine the **Materials and Methods** section of the paper.  How many marketing campaigns does this data represent?

The study was conducted over 2.5 years and covered 17 marketing campaigns.

### Problem 2: Read in the Data

Use pandas to read in the dataset `bank-additional-full.csv` and assign to a meaningful variable name.

### Imports

```python
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime

import random
```

Get the data
```Python
df = pd.read_csv('data/bank-additional-full.csv', sep = ';')
```

What does it look like?
```Python
df.head()
```

| age | job       | marital   | education   | default | housing | loan | contact   | month | day_of_week | ... | campaign | pdays | previous | poutcome    | emp.var.rate | cons.price.idx | cons.conf.idx | euribor3m | nr.employed | y   |
|-----|-----------|-----------|-------------|---------|---------|------|-----------|-------|-------------|-----|----------|-------|----------|-------------|--------------|----------------|---------------|-----------|-------------|-----|
| 56  | housemaid | married   | basic.4y    | no      | no      | no   | telephone | may   | mon         | ... | 1        | 999   | 0        | nonexistent | 1.1          | 93.994         | -36.4         | 4.857     | 5191.0      | no  |
| 57  | services  | married   | high.school | unknown | no      | no   | telephone | may   | mon         | ... | 1        | 999   | 0        | nonexistent | 1.1          | 93.994         | -36.4         | 4.857     | 5191.0      | no  |
| 37  | services  | married   | high.school | no      | yes     | no   | telephone | may   | mon         | ... | 1        | 999   | 0        | nonexistent | 1.1          | 93.994         | -36.4         | 4.857     | 5191.0      | no  |
| 40  | admin.    | married   | basic.6y    | no      | no      | no   | telephone | may   | mon         | ... | 1        | 999   | 0        | nonexistent | 1.1          | 93.994         | -36.4         | 4.857     | 5191.0      | no  |
| 56  | services  | married   | high.school | no      | no      | yes  | telephone | may   | mon         | ... | 1        | 999   | 0        | nonexistent | 1.1          | 93.994         | -36.4         | 4.857     | 5191.0      | no  |

### Problem 3: Understanding the Features

Examine the data description below, and determine if any of the features are missing values or need to be coerced to a different data type.

```Markdown
## Input variables:
#### Bank client data:
1. age (numeric)
2. job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3. marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4. education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5. default: has credit in default? (categorical: 'no','yes','unknown')
6. housing: has housing loan? (categorical: 'no','yes','unknown')
7. loan: has personal loan? (categorical: 'no','yes','unknown')
#### Related with the last contact of the current campaign:
8. contact: contact communication type (categorical: 'cellular','telephone')
9. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10. day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
#### Other attributes:
12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14. previous: number of contacts performed before this campaign and for this client (numeric)
15. poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
#### Social and economic context attributes
16. emp.var.rate: employment variation rate. quarterly indicator (numeric)
17. cons.price.idx: consumer price index. monthly indicator (numeric)
18. cons.conf.idx: consumer confidence index. monthly indicator (numeric)
19. euribor3m: euribor 3 month rate. daily indicator (numeric)
20. nr.employed: number of employees. quarterly indicator (numeric)

## Output variable (desired target):
21. y. has the client subscribed a term deposit? (binary: 'yes','no')
```

### Problem 4: Understanding the Task

After examining the description and data, your goal now is to clearly state the *Business Objective* of the task.  State the objective below.

```Python
df.info()
df.describe()
df.nunique()
print(df.iloc[random.randint(0, len(df))])
```
I needed to clean some of these categories up a bit:

```Python
df.rename(columns=lambda x: x.replace('.', '_'), inplace=True)
df.rename(columns={'y': 'response'}, inplace=True)
df['response'] = df['response'].map({'no': 0, 'yes': 1})
df['education'] = df['education'].str.replace('.', '_') 
```

### Problem 5: Engineering Features

Now that you understand your business objective, we will build a basic model to get started.  Before we can do this, we must work to encode the data.  Using just the bank information features (columns 1. 7), prepare the features and target column for modeling with appropriate encoding and transformations.

```Python
job_response_pct = pd.crosstab(df['response'], df['job']).apply(lambda x: x/x.sum() * 100)
job_response_pct = job_response_pct.transpose()
job_response_pct = df.groupby('job')['response'].mean() * 100

fig = go.Figure(data=[go.Bar(x=job_response_pct.index, y=job_response_pct.values)])
fig.update_layout(title='Subscription Rate by Job', xaxis_title='Job Category', yaxis_title='Subscription Rate')

fig.show()
```
![alt text](<images/Subscriptions by job.png>)
Then, I did this for marital status and education

```Python
married_response_percent = df.groupby('marital')['response'].mean() * 100
education_response_percent = df.groupby('education')['response'].mean() * 100

# Plotting job_response_pct, married_percent, and education_percent on a horizontal chart
fig = go.Figure()

fig.add_trace(go.Bar(y=job_response_pct.index, x=job_response_pct.values, name='Job', orientation='h'))
fig.add_trace(go.Bar(y=married_response_percent.index, x=married_response_percent.values, name='Marital', orientation='h'))
fig.add_trace(go.Bar(y=education_response_percent.index, x=education_response_percent.values, name='Education', orientation='h'))

fig.update_layout(title='Percentage Distribution of Job, Marital, and Education',
                  xaxis_title='Percentage',
                  yaxis_title='Categories')

fig.show()
```
![alt text](<images/Subscriptions by job_married_edu.png>)

Next, I needed to encode the categories to be able to run the classifiers

```Python
label_encoder = LabelEncoder()

for feature in df.columns:
    if df[feature].dtype == 'object':
        old_value = df[feature]
        df[feature] = label_encoder.fit_transform(df[feature])
        new_value = df[feature]
        print(f'{feature}: {dict(zip(new_value, old_value))}')
        
df.head()
```
```JSON
job: {3: 'housemaid', 7: 'services', 0: 'admin.', 1: 'blue-collar', 9: 'technician', 5: 'retired', 4: 'management', 10: 'unemployed', 6: 'self-employed', 11: 'unknown', 2: 'entrepreneur', 8: 'student'}
marital: {1: 'married', 2: 'single', 0: 'divorced', 3: 'unknown'}
education: {0: 'basic_4y', 3: 'high_school', 1: 'basic_6y', 2: 'basic_9y', 5: 'professional_course', 7: 'unknown', 6: 'university_degree', 4: 'illiterate'}
default: {0: 'no', 1: 'unknown', 2: 'yes'}
housing: {0: 'no', 2: 'yes', 1: 'unknown'}
loan: {0: 'no', 2: 'yes', 1: 'unknown'}
contact: {1: 'telephone', 0: 'cellular'}
month: {6: 'may', 4: 'jun', 3: 'jul', 1: 'aug', 8: 'oct', 7: 'nov', 2: 'dec', 5: 'mar', 0: 'apr', 9: 'sep'}
day_of_week: {1: 'mon', 3: 'tue', 4: 'wed', 2: 'thu', 0: 'fri'}
poutcome: {1: 'nonexistent', 0: 'failure', 2: 'success'}
```

Time to train the models

### Problem 6: Train/Test Split

With your data prepared, split it into a train and test set.
```Python
# split the data into training and testing sets

y = df['response']
X = df.drop('response', axis=1)
y = df['response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# check the shape of the training and testing sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
```

 ### Problem 7: A Baseline Model

Before we build our first model, we want to establish a baseline.  What is the baseline performance that our classifier should aim to beat?

```PYTHON
def train_model(model, X_train, y_train):
    start = datetime.now()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    end = datetime.now()
    elapsed = end - start
    return model, pred, elapsed.total_seconds()*1000.0

dummy = DummyClassifier(strategy='stratified', random_state=42)
dummy, dummy_pred, dummy_time = train_model(dummy, X_train, y_train)

dummy_test_accuracy = accuracy_score(y_test, dummy_pred)
dummy_train_accuracy = dummy.score(X_train, y_train)

performance_log = pd.DataFrame([['Dummy Classifier', dummy_time, dummy_test_accuracy, dummy_train_accuracy]],
                       columns=['Model', 'Training Time', 'Test Accuracy', 'Train Accuracy'])

print(performance_log)
print("=============================================")
print(f"Training time for Dummy Classifier: {dummy_time} milliseconds")
print(f"Accuracy of the model: {dummy_test_accuracy*100:.2f}%")
```

## I did this several more times each for the next 3 problems.
### Problem 8: A Simple Model
-- Use Logistic Regression to build a basic model on your data.  
### Problem 9: Score the Model
--What is the accuracy of your model?
### Problem 10: Model Comparisons
--Present your findings in a `DataFrame` similar to that below:

| Model | Train Time | Train Accuracy | Test Accuracy |
| ----- | ---------- | -------------  | -----------   |
|     |    |.     |.     |

```PYTHON
dt = DecisionTreeClassifier(random_state=42)
dt, dt_pred, dt_time = train_model(dt, X_train, y_train)

rf = RandomForestClassifier(random_state=42)
rf, rf_pred, rf_time = train_model(rf, X_train, y_train)

lr = LogisticRegression(random_state=42)
lr, lr_pred, lr_time = train_model(lr, X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn, knn_pred, knn_time = train_model(knn, X_train, y_train)

svm = SVC(random_state=42)
svm, svm_pred, svm_time = train_model(svm, X_train, y_train)
```

And got the final DatFrame showing the performance of each
```
                      Model  Training Time  Test Accuracy  Train Accuracy
0          Dummy Classifier          3.000       0.799466        0.800637
1  Decision Tree Classifier        166.833       0.889415        1.000000
2  Random Forest Classifier       4273.949       0.913328        1.000000
3       Logistic Regression        309.343       0.909808        0.908346
4       K-Nearest Neighbors        783.728       0.902282        0.931472
5    Support Vector Machine      19512.009       0.894513        0.898483
Training time for Support Vector Machine: 19512.009 milliseconds
Accuracy of the Support Vector Machine model: 89.45%
```

### Problem 11: Improving the Model

Now that we have some basic models on the board, we want to try to improve these.  Below, we list a few things to explore in this pursuit.

- More feature engineering and exploration.  For example, should we keep the gender feature?  Why or why not?
- Hyperparameter tuning and grid search.  All of our models have additional hyperparameters to tune and explore.  For example the number of neighbors in KNN or the maximum depth of a Decision Tree.  
- Adjust your performance metric

After running the previous models I wrote a method to randomly test whether a selected record in the data set would become a subscriber

```PYTHON
#Let's predict the outcome of a random record using all the models we have trained

def predict_with_model(model, record):
    name = type(model).__name__
    record = record.values.reshape(1, -1)
    prediction = model.predict(record)
    if prediction == 1:
        return f'The customer is predicted to subscribe, using {name} model'
    else:
        return f'The customer is not predicted to subscribe, using {name} model'

sample_record = X_test.iloc[random.randint(0, len(X_test))]
print(sample_record)

print(predict_with_model(dummy, sample_record))
print(predict_with_model(dt, sample_record))
print(predict_with_model(rf, sample_record))
print(predict_with_model(lr, sample_record))
print(predict_with_model(knn_pipe, sample_record))
print(predict_with_model(grid_search, sample_record))
print(predict_with_model(grid_search_cv, sample_record))
```
**Example Result**
```
age                 36.000
job                  0.000
marital              1.000
education            3.000
default              0.000
housing              0.000
loan                 0.000
contact              1.000
month                6.000
day_of_week          3.000
duration          1346.000
campaign             1.000
pdays              999.000
previous             0.000
poutcome             1.000
emp_var_rate        -1.800
cons_price_idx      92.893
cons_conf_idx      -46.200
euribor3m            1.266
nr_employed       5099.100
Name: 36052, dtype: float64
The customer is not predicted to subscribe, using DummyClassifier model
The customer is not predicted to subscribe, using DecisionTreeClassifier model
The customer is not predicted to subscribe, using RandomForestClassifier model
The customer is predicted to subscribe, using LogisticRegression model
The customer is predicted to subscribe, using Pipeline model
The customer is predicted to subscribe, using GridSearchCV model
The customer is predicted to subscribe, using GridSearchCV model
```
## So, what's the outcome?

Based on this exercise, the recommendations are as follows.

## Best model to use

The best model to use is the Logistic Regression model. It has a test accuracy of 0.91 and a training time of 309 milliseconds, although not the fastest. The training accuracy is 0.91, which is very high and does not signify overfitting like some of the other models. The model is not overfitting because the test accuracy is close to the training accuracy. The model is also not underfitting because the test accuracy is high.

## Second best model to use

If training time is not a significant concern then the Random Forest Classifier might be the best, although it's training accuracy of 1 is suspicious and indicates overfitting.

## Who should be the primary focus of future compaigns?

Based on the data, those most likely likely to subscribe are students & people with degrees, although there is a significant signal for those that are illiterate, but I'd want to dig into that more.

The second focus should also be on retired people as well.

There are other factors that could be derived as well, like age and whether they had a previous loan. But this needs further inquiry.

