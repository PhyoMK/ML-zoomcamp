import requests

url = 'http://localhost:9696/predict'

customer = {
 "gender": "female",
 "seniorcitizen": 0,
 "partner": "no",
 "dependents": "no",
 "tenure": 0,
 "phoneservice": "no",
 "multiplelines": "no_phone_service",
 "internetservice": "dsl",
 "onlinesecurity": "no",
 "onlinebackup": "no",
 "deviceprotection": "no",
 "techsupport": "no",
 "streamingtv": "no",
 "streamingmovies": "no",
 "contract": "month-to-month",
 "paperlessbilling": "yes",
 "paymentmethod": "mailed_check",
 "monthlycharges": 29.4,
 "totalcharges": 29.4
 }

response = requests.post(url, json = customer).json()
print(response)


if response['churn'] == True:
    print('sending promo email to %s' % ('9895-vfoxh'))
else:
    print('not sending promo email to %s' % ('9895-vfoxh'))







