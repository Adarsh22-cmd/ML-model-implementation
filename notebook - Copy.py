# TASK 4 : MACHINE LEARNING MODEL IMPLEMENTATION
# By adarsh sahu (CODTECH Intern)

# STEP 1 – Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# STEP 2 – Load Dataset
# Download dataset: https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep="\t")

print("Dataset Loaded")
data.head()

# STEP 3 – Preprocess Data
X = data['message']
y = data['label']

# Convert text to numeric using Bag-of-Words
cv = CountVectorizer()
X = cv.fit_transform(X)

# STEP 4 – Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5 – Build Model
model = MultinomialNB()

# STEP 6 – Train Model
model.fit(X_train, y_train)

# STEP 7 – Predict
y_pred = model.predict(X_test)

# STEP 8 – Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# STEP 9 – Test with custom message
msg = ["Congratulations! You won a cash prize"]
msg_vec = cv.transform(msg)
print("\nPrediction for sample message:", model.predict(msg_vec)[0])
