# Description: This code uses a basic Decision Tree Classifier to predict whether a Titanic passenger survived. I chose a decision tree because it's easy to understand and works well when you're just getting started with classification tasks.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Loading the data from the dataset file called titanicdisaster_dataset.
df = pd.read_excel(r"C:\Users\nmane\Documents\Machine Learning Projects\Titanic Disaster Project\titanicdisaster_dataset.xlsx")

# I picked a few columns that I thought were important for survival: passenger class, age, fare, and siblings/spouses onboard.
# I'm also keeping the passenger name to display results later.
data = df[['name', 'survived', 'pclass', 'age', 'fare', 'sibsp']]
data = data.dropna()  # Dropping rows with missing values so we don't get errors later.

# X has the inputs (features) and y has the output (label we're trying to predict)
X = data[['pclass', 'age', 'fare', 'sibsp']]
y = data['survived']

# Splitting the data so 70% is for training and 30% is for testing.
# This helps us see if the model actually works on new data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# I'm using a Decision Tree Classifier because it's beginner-friendly and doesn't need too much math to understand.
model = DecisionTreeClassifier()
model.fit(X_train, y_train)  # Train the model with the training data

# Now we use the model to make predictions based on the test data.
predictions = model.predict(X_test)

# We check how accurate the model is by comparing predictions to the actual answers.
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Let's look at a few example passengers to see what the model predicted.
# I used .loc here instead of .iloc to avoid the error from before.
data_test = data.loc[y_test.index].copy()
data_test['predicted'] = predictions

print("\nSample passengers and predictions:")
print(data_test[['name', 'pclass', 'age', 'fare', 'sibsp', 'survived', 'predicted']].head(10))