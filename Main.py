'''
Created on Feb 13, 2018

@author: MSURES56
'''

from flask import Flask
import os
from flask import jsonify
from sklearn import model_selection
from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pandas

app = Flask(__name__)
# api = Api(app)


@app.route('/')
def get():
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = pandas.read_csv(url, names=names)
	array = dataset.values
	X = array[:, 0:4]
	Y = array[:, 4]
	validation_size = 0.20
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)
	prediction = knn.predict(X_validation)
	asr = accuracy_score(Y_validation, prediction)
	#cf = confusion_matrix(Y_validation, prediction)
	cr = classification_report(Y_validation, prediction)
	pred = [{"accuracy_score":asr, "classification_report":cr, "algorithm_used":"KNeighborsClassifier"}]
	return jsonify(pred)


port = os.getenv('VCAP_APP_PORT', '5000')
if __name__ == "__main__":
	app.run(host='127.0.0.1', port=int(port))
	#app.run(host='0.0.0.0', port=int(port))
