from flask import Flask, request, jsonify, render_template
import pickle
import os
import json
import pandas as pd


app = Flask(__name__)
model = pickle.load(open(os.getcwd() + r'/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    req_dict = request.form.to_dict(flat=False)
    prediction = model.predict(pd.DataFrame.from_dict(req_dict))
    output = prediction[0]
    return render_template('index.html', prediction_text='Final destination is {}'.format(output))


@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict(pd.read_json(json.dumps(data), lines=True))

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)
