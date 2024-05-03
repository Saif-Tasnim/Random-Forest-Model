# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
data = pd.read_csv('../Datasets/transaction - used.csv')

# Step 1: Balance the 'isFraud' column using oversampling
oversample = RandomOverSampler(sampling_strategy='minority')
X_resampled, y_resampled = oversample.fit_resample(data.drop(columns=['isFraud']), data['isFraud'])

# Reduce the dataset to 20k entries using random sampling
data_resampled = pd.concat([X_resampled, y_resampled], axis=1)
data_resampled = data_resampled.sample(n=200000, random_state=42)

# Step 2: Plot a bar chart based on the resampled dataset
data_resampled['isFraud'].value_counts().plot(kind='bar')
plt.title('Distribution of isFraud')
plt.xlabel('isFraud')
plt.ylabel('Count')
plt.show()

# Step 3: Split the resampled dataset
X = data_resampled.drop(columns=['isFraud'])  # Features
y = data_resampled['isFraud']  # Target variable

# Convert categorical variables into numerical representations
X['type'] = pd.factorize(X['type'])[0]
X['nameOrig'] = pd.factorize(X['nameOrig'])[0]
X['nameDest'] = pd.factorize(X['nameDest'])[0]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Step 4: Run Random Forest model
start_time = time.time()

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Train the classifier
rf_classifier.fit(X_train, y_train)

end_time = time.time()

# Step 5: Print outputs
# Accuracy on training set
train_accuracy = rf_classifier.score(X_train, y_train)
print("Accuracy on Training Set:", train_accuracy)

# Accuracy on validation set
val_accuracy = rf_classifier.score(X_val, y_val)
print("Accuracy on Validation Set:", val_accuracy)

# Accuracy on testing set
test_accuracy = rf_classifier.score(X_test, y_test)
print("Accuracy on Testing Set:", test_accuracy)

# Step 6: Print total time taken
print("Total Time Taken:", end_time - start_time, "seconds")
