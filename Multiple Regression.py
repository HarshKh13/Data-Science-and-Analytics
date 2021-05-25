#Multiple Linear Regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

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