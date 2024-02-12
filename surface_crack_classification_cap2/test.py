import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
#url = 'https://lb2tezpjkl.execute-api.ap-southeast-1.amazonaws.com/test/predict'
data = {'url':'https://www.powerblanket.com/wp-content/uploads/2011/03/shutterstock_382144705.jpg'}

result = requests.post(url, json=data).json()
print(result)