#Chi square test of Independence
import pandas as pd
import numpy as np


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



























































































































