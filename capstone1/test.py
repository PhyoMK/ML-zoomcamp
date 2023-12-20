import requests

# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://lb2tezpjkl.execute-api.ap-southeast-1.amazonaws.com/test/predict'
data = {'url':'http://3.bp.blogspot.com/-ZXixpl4wbCE/UIXNI2jDBfI/AAAAAAAABDY/H8l5InQBgHI/s1600/Rottweiler-Puppy-Picture.JPG'}

result = requests.post(url, json=data).json()
print(result)

