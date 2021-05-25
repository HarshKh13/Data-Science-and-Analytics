#Categorical variable regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.pai as sm


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
