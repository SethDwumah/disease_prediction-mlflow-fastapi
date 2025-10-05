import requests


data ={ "text":"Fever, chills, headache, muscle aches, fatigue."}

res = requests.post("http://127.0.0.1:8802/predict",json=data)
print(res.json())