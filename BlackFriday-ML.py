import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings
from sklearn.neighbors import KNeighborsClassifier

# READ DATASET FROM LOCAL PATH
warnings.filterwarnings('ignore')
dataset = pd.read_csv(r'C:\Users\I am groot\PycharmProjects\test\input\data\blackfriday.csv')
print(dataset.head(20))

# PEEK THE DATASET & VIEW THE GROUPED DATA
print(dataset.shape)
print(dataset.describe())
print(dataset.groupby('Product_Category').size())

# PLOT THE DATA FOR VISUALISATION ON GRAPH
sns.regplot(x=dataset["Product_Category"], y=dataset["Purchase"])
plt.show()
sns.lmplot('Purchase', 'Occupation', dataset, hue='Product_Category', fit_reg=False)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()

# DIVIDE THE DATA IN TRAINING AND TESTING SETS
array = dataset.values
X = array[:, 3:6]
Y = array[:, 6]
Y = Y.astype('int')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

print("x_train")
print(X_train)
print("y_train")
print(Y_train)

# FIT DATA TO DIFFERENT MODELS & VIEW THE RESULT
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
Predictions = model.predict(X_test)
print(accuracy_score(Y_test, Predictions))

model = RandomForestClassifier(n_estimators=5)
model.fit(X_train, Y_train)
Predictions = model.predict(X_test)
print(accuracy_score(Y_test, Predictions))

model = LogisticRegression()
model.fit(X_train, Y_train)
Predictions = model.predict(X_test)
print(accuracy_score(Y_test, Predictions))

# THE BEST RESULT IS OBSERVERED FROM THE KNN MODEL

# PRINT THE CONFUSION MATRIX AND CLASSIFICATION REPORT
print("--------CONFUSION MATRIX---------")
print(confusion_matrix(Y_test, Predictions))
print(classification_report(Y_test, Predictions))
