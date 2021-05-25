#Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 
df = pd.read_csv(path)
x = df.iloc[:,;-1].values
y = df.iloc[:,-1].values
x = sm.add_constant(x)
logistitc_model = sm.Logit(y,x)
result = logistic_model.fit()
result.summary()

print(model.coef_)

#Confusion matrix
path = 
df = pd.read_excel(path,engine = 'openpyxl')
df.head()
df.describe()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegession
x = df[['Spending','Card']]
y = df['Coupon'].values.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)

log_reg = LogisticRegression()
log_reg.fit(x_train,y_train.ravel())
y_pred = log_reg.predict(x_test)
print(y_pred)

y_prob_train = log_reg.predict_proba(x_train)[:,1]
y_prob_train.reshape(1,-1)
print(y_prob_train)

y_prob_test = log_reg.predict_proba(x_test)[:,1]
y_prob_test.reshape(1,-1)
print(y_prob_test)

x = sm.add_constant(x)
logit_model = sm.Logit(y,x)
result = logit_model.fit()
result.summary()

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)
print(score)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

true_negetives,false_positives,false_negetives,true_positives = 
confusion_matrix(y_test,y_pred).ravel()

print("True Positives",true_positives)
print("True negetives",true_negetives)
print("False Positives",false_positives)
print("False negetives",false_negetives)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))

accuracy = (true_positives+true_negitives)/(true_positives+true_negetives+
                                            false_positives+false_negetives)
print("Accuracy {0.2f}".format(accuracy))
specificity = true_negetives/(true_negetives + false_positives)
print("Specificity {0.2f}".format(specificity))
sensitivity = true_positives/(true_positives + false_negetives)
print("Sensitivity {0.2f}".format(sensitivity))

#ROC curve
from sklearn.metrics import roc_auc_curve
from sklearn.metrics import roc_curve, auc
fpr,tpr,thresholds = roc_curve(y_train,y_prob_train)
roc_auc = auc(fpr,tpr)

plt.figure()
plt.plot(fpr,tpr,color='blue',label = 'ROC curve (area = %0.2f)'%roc_auc)
plt.plot([0,0],[1,1],color = 'red')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title("ROC curve for training data")
plt.legend(loc = 'lower right')
plt.show()

fpr,tpr,thresholds = roc_curve(y_test,y_prob_test)
roc_auc = auc(fpr,tpr)

plt.figure()
plt.plot(fpr,tpr,color='blue',label = 'ROC curve (area = %0.2f)'%roc_auc)
plt.plot([0,0],[1,1],color = 'red')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title("ROC curve for training data")
plt.legend(loc = 'lower right')
plt.show()

fpr,tpr,thresholds = roc_curve(y_test,y_prob_test)
indices = np.arange(len(fpr))
df_roc = pd.DataFrame({'fpr':pd.Series(fpr,indices),'tpr':pd.Series(tpr,indices),
                       '1-fpr':pd.Series(1-fpr,indices),
                       'tf':pd.Series(tpr-(1-fpr),indices),
                       'thresholds':pd.Series(thresholds,indices)})

df_roc.iloc[(df_roc.tf-0).abs().argsort()[:1]]

fig,ax = plt.subplots()
plt.plot(df['tpr'],color = 'blue')
plt.plot(df['1-fpr'],color = 'red')
plt.xlabel('1-False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()


#Classification using optimal threshold value
from sklearn.preprocessing import binarize
opt_thresh = 
y_pred = binarize(y_prob_test.reshape(1,-1), opt_thresh)[0]
print(y_pred)

confusion_mat = confusion_matrix(y_test,y_pred)
print(confusion_mat)

classification_repo = classification_report(y_test,y_pred)
print(classification_repo)