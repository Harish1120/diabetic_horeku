import joblib

# Load the model

model = joblib.load('diabetic_80.pkl')

# prediction
data = model.predict([[0,1,1,1,1,85,5,60]])

if data == 0:
    print('Patient is not diabetic')
else:
    print('Patient is diabetic')