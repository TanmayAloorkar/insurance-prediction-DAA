import numpy as mp
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

df= pd.read_csv("insurance_data.csv")
df.head()

#train and test split function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age','affordibility']] ,df.bought_insurance ,test_size=0.2,  random_state=25)

X_train.head()
X_test.head()


len(X_train)

#Scalling
x_train_Scalled=X_train.copy()
x_train_Scalled['age']=x_train_Scalled['age']/100
x_test_Scalled=X_test.copy()
x_test_Scalled['age']=x_test_Scalled['age']/100
x_train_Scalled.head()

#Keras
model=keras.Sequential([
    keras.layers.Dense(1,input_shape=(2,),activation='sigmoid', kernel_initializer='ones',bias_initializer='zeros')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train_Scalled,y_train,epochs=5000)

model.evaluate(x_test_Scalled,y_test)

model.predict(x_test_Scalled)
x_test_Scalled
model.get_weights()

import matplotlib.pyplot as plt

# Assuming you have trained your model and made predictions
predictions = model.predict(x_test_Scalled)

plt.figure(figsize=(8, 6))
plt.scatter(x_test_Scalled['age'], y_test, color='blue', label='Actual')
plt.scatter(x_test_Scalled['age'], predictions, color='red', label='Predicted')
plt.xlabel('Age (Scaled)')
plt.ylabel('Bought Insurance')
plt.title('Scatter Plot of Predictions vs. Actual')
plt.legend()
plt.show()
