import os
os.system('color 3f')
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import win32com.client as wincl
speak = wincl.Dispatch("SAPI.SpVoice")

data = pd.read_csv('diabetes.csv')

x = data.iloc[:,[1,2,3,4,5,6,7]]
y = data.iloc[:,[8]]

model = KNeighborsClassifier()

model.fit(x,y)

print('WelCome to Diabetes Prediction Software') #greeting
speak.Speak('WelCome to Diabetes Prediction Software')


print("Enter Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age 'with comma'")
speak.Speak("Enter Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age 'with comma'")
val = list(input().split(","))

pred = model.predict([val])
print(pred)

if pred == [1]:
    print('You Have Diabetes')
    speak.Speak('You Have Diabetes')
    print('You need to take this medicines')
    speak.Speak('You need to take this medicines')
    print('Alpha-glucosidase inhibitors, Biguanides, Dopamine agonist, DPP-4 inhibitors, Meglitinides')
    speak.Speak('Alpha-glucosidase inhibitors, Biguanides, Dopamine agonist, DPP-4 inhibitors, Meglitinides')
    print('And immediate contact to your doctor')
    speak.Speak('and immediate contact to your doctor')

else:
    print('You have not Diabetes')
    speak.Speak('You have not Diabetes')