#2 sample
#Population mean test, sd known
import numpy as np
from scipy import stats
import pandas as pd

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
critical_val = stats.t.ppf(alpha/2,dof)
print(critical_val)

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
print(critical_val)


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