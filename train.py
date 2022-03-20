import pandas as pd 
from sklearn import model_selection 
from sklearn.linear_model import LogisticRegression 
import joblib

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

name = ['preg','plas','pres','skin','test','mass','pedi','age','class']

df = pd.read_csv(url,names = name)
print(df.head())

array = df.values

x,y = array[:,0:8], array[:,8]

x_train,x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size = 0.2, random_state=101)

#training model 
model = LogisticRegression()
model.fit(x_train,y_train)
print('[Info] - model has been trained')

# accuracy
result = model.score(x_test,y_test)
print(f'[INFO] - model accuracy is {result}')

#model saving
filename = 'diabetic_80.pkl' #.pkl or .sav
joblib.dump(model , filename)


