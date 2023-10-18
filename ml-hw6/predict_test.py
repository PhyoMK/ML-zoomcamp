import requests

url = 'http://localhost:9696/credict'
client = {"job": "retired", "duration": 445, "poutcome": "success"}
response = requests.post(url, json = client).json()
print(response)



