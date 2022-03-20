from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route('/')
def home(): 
    return render_template('home.html')

@app.route('/predict',methods=["post"])
def predict():
    preg = request.form.get('preg')
    plas = request.form.get('plas')
    pres = request.form.get('pres')
    skin = request.form.get('skin')
    test = request.form.get('test')
    mass = request.form.get('mass')
    pedi = request.form.get('pedi')
    age = request.form.get('age')


    print(preg,plas,pres,skin,test,mass,pedi,age)
 # Load the model

    model = joblib.load('diabetic_80.pkl')
# prediction
    data = model.predict([[preg,plas,pres,skin,test,mass,pedi,age]])

    if data == 0:
        output  = 'Patient is not diabetic'
    else:
        output = 'Patient is diabetic'

    return render_template('predict.html', output = output)
    
if __name__ == '__main__':
    app.run(debug = True)   