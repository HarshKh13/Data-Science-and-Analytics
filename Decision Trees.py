import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

path = 
data = pd.read_excel(path)
data.head()

lab_encod = LabelEncoder()
age_encod = lab_encod.fit_transform(data['age'])
income_encod = lab_encod.fit_transform(data['income'])
student_encod = lab_encod.fit_transform(data['student'])
credit_rating_encod = lab_encod.fit_transform(data['credit_rating'])
buys_comp_encod = lab_encod.fit_transform(data['buys_computer'])

data.head()

data_new = data.drop(['age','income','student','credit_rating',
                      'buys_computer'],axis='columns')
data_new.head()

feature_cols = ['age_encod','income_encod','student_encod',
                'credit_rating_encod']
x = data_new.drop['buys_comp_encod','RID']
y = data_new['buys_comp_encod']
x.head()
y.head()

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
result = dtc.fit(x,y)
print(result)

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(result,out_file = dot_data,
                filled=True, rounded= True,
                special_characters = True,feature_names = feature_cols,
                class_names = ['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('buys_computer.png')

Image(graph.create_png())

#Model Evaluation
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
clf = DecisionTreeClassifier()
result = clf.fit(x_train,y_train)
print(result)

from sklearn.metrics import accuracy_score
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)


















