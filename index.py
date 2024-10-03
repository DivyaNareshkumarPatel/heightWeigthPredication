from flask import Flask, render_template, url_for, request
import pandas as pandas
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    with open('models\model.pkl', 'rb') as f1:
        clf = pickle.load(f1)

        if request.method == 'POST':
            pname = float(request.form['pname'])
            input_data = [[pname]]
            pred = clf.predict(input_data)
        return render_template('result.html', prediction=pred, name=pname)

if __name__ == '__main__':
    app.run(debug=True)