import numpy as np
from scipy import stats
import pandas as pd

#1 sample
#Test of population mean, Population sd known
#one-tailed test
def cal_z_score_1samp(sample_mean,pop_mean,sample_size,
                      pop_sd):
    sample_sd = pop_sd/np.sqrt(sample_size)
    z_score = (sample_mean-pop_mean)/sample_sd
    return z_score

def cal_p_value(z_value):
    if z_value <= 0:
        p_val = stats.norm.cdf(z_value)
    else:
        p_val = 1 - stats.norm.cdf(z_value)
        
    return p_val

def cal_critical_value(confidence,bool flag):
    if flag==True:
        critical_val = stats.norm.ppf(confidence)
    else:
        critical_val = stats.norm.ppf(1-condfidence)
    
    return critical_val

sample_mean = 
pop_mean = 
pop_sd = 
sample_size = 
alpha = 
flag = 

#P-value approach
z_val = cal_z_score_1samp(sample_mean,pop_mean,sample_size,
                          pop_sd)
print("Z-value",z_val)
p_val = cal_p_value(z_val)
print("P-value",p_val)
if p_value < alpha:
    print("We reject null hypothesis")
else:
    print("We do not reject null hypothesis")

#Critical value approach
critical_val = cal_critical_value(alpha,flag)

#Test of population mean
#Two-tail test
sample_mean = 
pop_mean = 
pop_sd = 
sample_size = 
alpha = 

#p-value approach
z_val = cal_z_score_1samp(sample_mean,pop_mean,sample_size,
                          pop_sd)
print("Z-value",z_val)
p_val = cal_p_val(z_val)
print("P-value",p_val)
if p_value < alpha/2:
    print("We reject null hypothesis")
else:
    print("We do not reject null hypothesis")

#Critical-value approach
critical_val = cal_critical_value(alpha/2,flag = True)

#Test of population mean, Population sd unknown
path = 'icecream sale data.xlsx'
df_icecream = pd.read_excel(path,engine = 'openpyxl')
def cal_z_value_1samp_sd_unknown(sample_mean,pop_mean,sample_sd,
                                 sample_size):
    combined_sd = sample_sd/np.sqrt(sample_size)
    z_val = (sample_mean-pop_mean)/combined_sd
    return z_val

x = df_icecream['Number of ice cream sold']
sample_mean = np.mean(x)
sample_sd = np.std(x)
sample_size = len(x)
dof = sample_size-1
alpha = 0.05
pop_mean = 10
z_val = cal_z_val_1samp_sd_unknown(sample_mean,pop_mean,sample_sd,
                                   sample_size)
print("Z-value",z_val)
p_val = cal_p_value(z_val)
print("P-value",p_val)

#direct method
stats.ttest_1samp(x,pop_mean)
critical_z_value = stats.t.ppf(alpha,dof)
print(critical_z_value)


#Population proportion test
def cal_z_value_1samp_prop(sample_prop,pop_prop,
                               sample_size):
    sample_sd = np.sqrt(pop_prop*(1-pop_prop)/sample_size)
    z_val = (sample_prop-pop_prop)/sample_sd
    return z_val

count = 
sample_size = 
sample_prop = count/sample_size
pop_prop = 
alpha = 

z_val = cal_z_value_1samp_prop(sample_prop,pop_prop,sample_size)
print("Z-value",z_val)
p_val = cal_p_value(z_val)
print("P-value",p_val)

#direct method
from statsmodels.stats.proportion import proportions_ztest
proportions_ztest(count,sample_size,alpha)

#2 sample
#Population mean test, sd known
def cal_z_value_2samp_sdknown(sample1_size,sample2_size,pop1_sd,
                            pop2_sd,sample1_size,sample2_size):
    pop1_var = pop1_sd**2
    pop2_var = pop2_sd**2
    combined_sd = np.sqrt(pop1_var/sample1_size + pop2_var/sample2_size)
    z_val = (sample1_mean-sample2_mean)/combined_sd
    return sd

sample1_mean = 
sample2_mean = 
pop1_sd = 
pop2_sd = 
sample1_size = 
sample2_size = 
z_val = cal_z_value_2samp_sdknown(sample1_mean,sample2_mean,pop1_sd,
                                  pop2_sd,sample1_size,sample2_size):
print("z-value",z_val)
p_val = cal_p_value(z_val)
print("P-value",p_val)

#population mean test, sd unknown
#sd assumed equal
def cal_z_value_2samp_sdunknownequal(sample1_mean,sample2_size,sample1_sd,
                                sample2_sd,sample1_size,sample2_size,dof):
    sample1_var = sample1_sd**2
    sample2_var = sample2_sd**2
    sample_var = ((sample1_size-1)*sample1_var + (sample2_size-1)*sample2_var)
    sample_var = sample_var/dof
    sample_sd = np.sqrt(sample_var)
    combined_sd = sample_sd*np.sqrt(1/sample1_size + 1/sample2_size)
    t_val = (sample1_mean - sample2_mine)/combined_sd
    return t_val

a = []
b = []
sample1_mean = np.mean(a)
sample2_mean = np.mean(b)
sample1_sd = np.std(a)
sample2_sd = np.std(b)
sample1_size = len(a)
sample2_size = len(b)
dof = sample1_size + sample2_size - 2
alpha = 0.05

#p-value approach
z_val = cal_z_value_2samp_sdunknownequal(sample1_mean,sample2_mean,sample1_sd,
                                         sample2_sd,sample1_size,sample2_size)
print("Z-value",z_val)
p_val = cal_p_value(z_val)
print("P-value",p_val)

#direct method
stats.ttest_ind(a,b,equal_var = True)

#critical value method
stats.t.ppf(alpha/2,dof)

#population mean test, sd unknown
#assumed unequal
def cal_z_value_2samp_sdunknown_unequal(sample1_mean,sample2_mean,
                                        sample1_sd,sample2_sd,sample1_size,
                                        sample2_size):
    sample1_var = sample1_sd**2
    sample2_var = sample2_sd**2
    combined_sd = np.sqrt(sample1_var/sample1_size + sample2_size/sample2_size)
    z_val = (sample1_mean - sample2_mean)/combined_sd
    return z_val

def cal_dof_2samp_sdunknown_unequal(sample1_sd,sample2_sd, sample1_size,
                                    sample2_size):
    sample1_var = sample1_sd**2
    sample2_var = sample2_sd**2
    deno = (sample1_var/sample1_size)**2/(sample1_size-1)
    deno = deno + (sample2_var/sample2_size)**2/(sample2_size-1)
    num = (sample1_var/sample1_size) + (sample2_var/sample2_size)
    num = num**2
    dof = num//deno
    return dof

a = 
b = 
sample1_mean = np.mean(a)
sample2_mean = np.mean(b)
sample1_sd = np.std(a)
sample2_sd = np.std(b)
sample1_size = len(a)
sample2_size = len(b)

#p-value approach
z_val = cal_z_value_2samp_sdunknown_unequal(sample1_mean,sample2_mean,
                                        sample1_sd,sample2_sd,sample1_size,
                                        sample2_size)
print("z-value",z_val)
p_val = cal_p_value(z_val)
print("Z-value",z_val)

#direct method
stats.ttest_ind(a,b,equal_var = False)

#critical value approach
dof = cal_dof_2samp_sdunknown_unequal(sample1_sd,sample2_sd, sample1_size,
                                    sample2_size)
critical_value = stats.t.ppf(alpha/2,dof)


# 2 sample population proportion
def cal_z_value_2samp_prop(sample1_prop,sample2_prop,sample1_size,
                           sample2_size):
    p_bar = sample1_size*sample1_prop + sample2_size*sample2_prop
    combined_sd = p_bar*(1-p_bar)*(1/sample1_size + 1/sample2_size)
    combined_sd = np.sqrt(combined_sd)
    z_val = (sample1_prop-sample2_prop)/combined_sd
    return z_val

sample1_prop = 
sample2_prop = 
sample1_size = 
sample2_size = 

#p-value approach
z_val = cal_z_val_2samp_prop(sample1_prop,sample2_prop,sample1_size,
                             sample2_size)
print("Z-value",z_val)
p_val = cal_p_val(z_val)
print("P-value",p_val)

#Linear regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
path = 
df = pd.read_csv(path)
x = df['TV ads'].values
y = df['car Sold'].values
plt.plot(x,y,'o')
plt.xlabel('TV ads')
plt.ylabel('Cars sold')
plt.title('Cars sold vs tv ads')
plt.show()

x = sm.add_constant(x)
linear_model = sm.OLS(y,x)
result = linear_model.fit()
result.summary()

path = 
df = pd.read_csv(path)
x = df['Population'].values
y = df['Sales'].values
plt.plot(x,y,'o')
plt.xlabel('Population')
plt.ylabel('Icecream sales')
plt.title('Sales vs population')
plt.show()

x = sm.add_constant(x)
linear_model = sm.OLS(y,x)
result = linear_model.fit()
result.summary()


plt.figure()
sns.regplot(x,y,fit_reg=True)
plt.scatter(np.mean(x),np.mean(y),color='red')

from statsmodels.stats.ouliers_infulence import summary_table
s1, data, s2 = summary_table(result,alpha = 0.05)
y_pred = data[:,2]
confidence_int_low,confidence_int_high = data[:,4:6].T
pred_int_low,pred_int_high = data[:,6:8].T

print(confidence_int_low)
print(confidence_int_high)
print(pred_int_low)
print(pred_int_high)

fig,ax = plt.subplots(figsie = (8,6))
ax.plot(x,y,'o',label = 'data')
ax.plot(x,y_pred,'r-',label = 'OLS')
ax.plot(x,confidence_int_low,'g--')
ax.plot(x,confidence_int_high,'g--')
ax.plot(x,pred_int_low,'b--')
ax.plot(x,pred_int_high,'b--')
plt.show()


#Multiple Linear Regression
path = 
df = pd.read_csv(path)
x1 = df['x1'].values
x2 = df['n_of_deliveries'].values
y = df['travel_time']
plt.scatter(x1,y,color='blue')
plt.xlabel('Miles Travelled')
plt.ylabel('Travel time')
plt.title('Regression plot of travel time against miles travelled')
plt.show() 

plt.scatter(x2,y,color='blue')
plt.xlabel('No of deliveries')
plt.ylabel('Travel time')
plt.title('Regression plot of travel time against number of deliveries')  
plt.show()

plt.figure()
plt.scatter(x1,y,color='red')
plt.scatter(x2,y,color='blue')
plt.xlabel('x1 in red and x2 in blue')
plt.ylabel('travel time')
plt.title('Multiple regression')
plt.show()

linear_reg1 = sm.OLS(formula = "travel time ~ x1",data = df)
result = linear_reg1.fit()
result.summary()

linear_reg2 = sm.OLS(formula = "travel time ~ n_of_deliveries",data = df)
result = linear_reg2.fit()
result.summary()

multiple_reg = sm.OLS(formula = "travel time ~ x1+n_of_deliveries",data = df)
result = multiple_reg.fit()
result.summary()

#Categorical variable regression
path = 
df = pd.read_csv(path)
df.head()

x1 = df['months_since_last_service'].values
x2 = df['type_of_repair']
y = df['repair_time_in_hours'].values
plt.scatter(x1,y,color = 'green')
plt.xlabel('Months size last service')
plt.ylabel('Repair time in hours')
plt.title('Regression plot of repair time vs months since last service')
plt.show()

dummy_var = pd.get_dummies(x2)
dummy_var.head()

df = pd.concat([df,dummy_var],axis=1)
df.drop(['type_of_repair','mechanical'],axis=1)

x = df[['months_since_last_service','electrical']]
model = sm.OLS(y,x)
result = model.fit()
result.summary()

#Logistic Regression
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

#Chi square test of Independence
path = 
df_acad = pd.read_csv(path)
df_acad.head()

#Chi square test of independence between Gender and Student Motivation
df_hypo = pd.pivot_table(df_acad[['g','sm']],index = 'g',columns = 'sm',
                         aggfunc = len)
df_hypo.head()

from scipy.stats import chi2_contigency
chi_val,p_val,dof,contigency_tbl = chi2_contigency(df_hypo)
print("Chi-value",chi_val)
print("P-value",p_val)
print("Degrees of Freedom",dof)

print(contigency_tbl)

#Chi square goodness of fit test
import scipy 
from scipy.stats import chi2
from scipy.stats import poisson
from scipt.stats import chisquare

import pandas as pd
import numpy as np
path = 
data = pd.read_excel(path)
data.head()

observed_freq = data['freuqency']
total_arrivals = np.sum(data['Arrivals']*data['frequency'])
total_frequency = np.sum(data['frequency'])
avg = total_arrivals/total_frequency

expected_freq = []
for i in range(len(observed_freq)):
    poisson_val = poisson.pmf(i,avg)
    temp_exp_val = total_frequency*poisson_val
    expected_freq.append(temp_exp_val)

print(expected_freq)

expected_freq = [round(itr,2) for itr in expected_freq]
print(expected_freq)

indices = np.arange(len(observed_freq))
data_chisq = pd.DataFrame({'Observed freq':pd.Series(observed_freq,indices)
                           ,'Expected_freq':pd.Series(expected_freq,indices)})
df_chisq.head()

chisquare(observed_freq,expected_freq)



























































































































