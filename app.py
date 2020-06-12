from flask import Flask,render_template,request,jsonify
from flask_cors import CORS
import pickle
import joblib
import base64


app=Flask(__name__)

CORS(app)

@app.route('/')
def homepage():

    return '<h1> Server Started </h1>'

@app.route('/gui')
def gui():
    return render_template('gui.html')


@app.route('/analyse',methods=['GET','POST'])
def analyse():
    print(request.method)
    msg=(request.args.get('data'))

    prediction=classify(msg)

    resp={'pred':prediction}

    return jsonify(resp)


def classify(msg):
    msg1 = [msg]

    cv = pickle.load(open('transformer.pkl', 'rb'))
    clsfr = joblib.load('Spam-Classifier.pkl')

    vect = cv.transform(msg1).toarray()

    ind=(clsfr.predict(vect)[0])


    li=['HAM','SPAM']

    return li[ind]
#"#003366"










if __name__ == '__main__':
    app.run()
