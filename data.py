import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import tkinter as tk
from tkinter import Label, Entry, Button, Text, PhotoImage
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
import pickle

data = pd.read_csv("/Users/Saumya/Downloads/air.csv")

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

def predict_pm25():
    input_date = date_entry.get()

    try:
        input_date = datetime.strptime(input_date, '%Y-%m-%d %H:00')
    except ValueError:
        result_text.config(state=tk.NORMAL)
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Invalid date format. Please use 'YYYY-MM-DD HH:00' format.")
        result_text.config(state=tk.DISABLED)
        return

    future_features = [input_date.year, input_date.month, input_date.day, input_date.hour]

    predicted_pm25 = model.predict([future_features])

    pm25_category = categorize_pm25(predicted_pm25[0])

    result_text.config(state=tk.NORMAL)
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, f'Predicted PM2.5: {predicted_pm25[0]:.2f}\nCategory: {pm25_category}')
    result_text.config(state=tk.DISABLED)

def categorize_pm25(pm25_value):
    if pm25_value <= 50:
        return "Good"
    elif pm25_value <= 100:
        return "Moderate"
    elif pm25_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif pm25_value <= 200:
        return "Unhealthy"
    elif pm25_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

window = ThemedTk(theme='sun-valley')
window.title("PM2.5 Prediction")

background_image = Image.open("/Users/Saumya/Downloads/WhatsApp Image 2023-11-06 at 4.03.37 PM.jpeg")  
background_image = ImageTk.PhotoImage(background_image)
background_label = Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

date_label = Label(window, text="Enter a future date in the format 'YYYY-MM-DD HH:00':")
date_label.grid(row=0, column=0, columnspan=4, pady=10)
date_entry = Entry(window)
date_entry.grid(row=0, column=4, columnspan=4,padx =5)

predict_button = Button(window, text="Predict PM2.5", command=predict_pm25)
predict_button.grid(row=0, column=8, columnspan=2,padx =5)

result_text = Text(window, height=2, width=40)
result_text.config(state=tk.DISABLED)
result_text.grid(row=1, column=0, columnspan=10, pady=10)

window.mainloop()