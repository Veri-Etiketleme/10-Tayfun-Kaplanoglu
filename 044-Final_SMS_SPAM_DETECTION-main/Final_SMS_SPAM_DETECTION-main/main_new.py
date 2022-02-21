# import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.pipeline import Pipeline
import pickle

app = Flask(__name__,static_url_path='/templates/',static_folder='templates')
model = pickle.load(open('C:\\Users\\Aniket Ratnaparakhi\\PycharmProjects\\SMS_SPAM_NEW\\finalized_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():

    final_feature = request.form.values()
    prediction = model.predict(final_feature)
    result = ['WAIT A MINUTE, This IS A SPAM!', 'OHHH, THIS IS A NORMAL MESSAGE.']
    # output = (prediction[0], 2)
    # def pre():
    if prediction:
        out=result[0]
    else:
        out=result[1]

    return render_template('index1.html', prediction_text=out)

@app.route('/results',methods=['POST'])
def results():

    data = str(request.get_json(force=True))
    # prediction = model.predict(data.get())
    prediction = model.predict(data['msg'])
    output = prediction
    return jsonify(output)
# @app.route("/profile/<string:name>")
# def profile(name):
#   return render_template("test.html", name=name)

if __name__ == "__main__":
    app.run(debug=True,port=5000)
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=5000)