import requests

host = 'finalcheck-heartdisease-env.eba-sr8kwj2b.ap-southeast-1.elasticbeanstalk.com'
url = f'http://{host}/predict'

customer = {
     "age": 56,
     "sex": "female",
     "chest_pain_type": "asymptomatic",
     "resting_blood_pressure": 150,
     "cholesterol": 230,
     "fasting_blood_sugar": "cls",
     "resting_ecg": "st_t_wave_abnormal",
     "max_heart_rate": 124,
     "exercise_angina": "yes",
     "oldpeak": 1.5,
     "st_slope": "flat"
}

response = requests.post(url, json = customer).json()
print(response)


if response['heartdisease'] == True:
    print('The person has heart disease')
else:
    print('The person has no heart disease')

