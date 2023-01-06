from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

loan_data = pd.read_csv('Loan_Data.csv')

X = pd.get_dummies(loan_data.drop(['Loan_Status', 'Loan_ID'], axis=1))
y = loan_data['Loan_Status'].apply(lambda x: 1 if x=='Y' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

model = Sequential()
model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

model.fit(X_train, y_train, epochs=500, batch_size=32)

y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]

print(accuracy_score(y_test, y_hat))
