import flask
from flask import Flask,request, render_template
#from flask_cors import CORS
import joblib




app = Flask(__name__)


@app.route('/', methods=['GET'])
def sendHomePage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predictPerformance():
    cylinders = float(request.form['cylinders'])
    displacement = float(request.form['displacement'])
    horsepower = float(request.form['horsepower'])
    weight = float(request.form['weight'])
    acceleration = float(request.form['acceleration'])
    modelyear = float(request.form['modelyear'])
    origin = float(request.form['origin'])

    X = [[cylinders,displacement,horsepower,
          weight,acceleration,modelyear,origin]]

    model = joblib.load('gbr_performance.pkl')

    rating = model.predict(X)[0]
    return render_template('predict.html',predict=rating)

# if __name__ == '__main__':
#     app.run()
if __name__ == '__main__':
    # app.debug = True
    app.run()
    
