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