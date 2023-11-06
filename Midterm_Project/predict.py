import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_rf.bin'

with open(model_file, 'rb') as f_in:
    dv, rf = pickle.load(f_in)

app = Flask('heartdisease')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = rf.predict_proba(X)[0, 1]
    heart_disease= y_pred >= 0.5

    result = {
        'heart disease probability': float(y_pred),
        'heartdisease': bool(heart_disease)
    }

    return jsonify(result)

if __name__ == "__main__":
     app.run(debug=True, host='0.0.0.0', port=9696)
