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
print(critical_val)

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
print(critical_val)

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

