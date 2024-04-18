import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/Saumya/Downloads/air.csv')

data['Timestamp'] = pd.to_datetime(data['Timestamp'])

data['Year'] = data['Timestamp'].dt.year
data['Month'] = data['Timestamp'].dt.month
data['Day'] = data['Timestamp'].dt.day
data['Hour'] = data['Timestamp'].dt.hour

features = ['Year', 'Month', 'Day', 'Hour']
X = data[features]
y = data['PM2.5']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestRegressor(n_estimators=300, max_depth=30, random_state=0)

model.fit(X_train, y_train)

data_2022 = data[data['Year'] == 2022]

y_predicted_2022 = model.predict(data_2022[features])

r2=r2_score(y_true=data_2022['PM2.5'],y_pred=y_predicted_2022)
print(r2) #97

plt.figure(figsize=(12, 6))
plt.plot(data_2022['Timestamp'], data_2022['PM2.5'], label='Original PM2.5', color='blue')
plt.plot(data_2022['Timestamp'], y_predicted_2022, label='Predicted PM2.5', color='red')
plt.xlabel('Date')
plt.ylabel('PM2.5 Level')
plt.title('Original vs. Predicted PM2.5 Levels for Year 2022')
plt.legend()
plt.grid(True)
plt.show()