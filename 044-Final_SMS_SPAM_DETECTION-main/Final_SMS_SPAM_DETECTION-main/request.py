import requests
import pickle
url = 'http://localhost:5000/results'
msg1=['Hurry up free mobile shop now!']
r = requests.post(url,json={'msg':msg1})

print(r.json().str)

# model = pickle.load(open('finalized_model.pkl','rb'))
# model.predict(["hi aniket i call u back"])
# requests.post(url,data=)