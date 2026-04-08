import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
df = pd.read_csv('C:\\Users\\nivas\\OneDrive\\Desktop\\sales\\sales_data_sample.csv', encoding='latin1')
#print(df.head())

df=df.drop(['ADDRESSLINE2'],axis=1)
df=df.fillna('0')
#print(df.info())
#print(df.isna().sum())
df['ORDERDATE']=pd.to_datetime(df['ORDERDATE'])
df['MONTH']=df['ORDERDATE'].dt.month
df['YEAR']=df['ORDERDATE'].dt.year
#TOTAL SALES
total_sales=df['SALES'].sum()
print(total_sales)


#monthly wise sales

monthly=df.groupby('MONTH')["SALES"].sum()
print(monthly)

#year sales

yearly=df.groupby('YEAR')["SALES"].sum()

print(yearly)

#total product 

product=df.groupby('PRODUCTLINE')['SALES'].sum()
print(product)

#CUSTOMER
cust=df.groupby('CUSTOMERNAME')['SALES'].sum()
print(cust)


'''df.plot(xlabel='PRODUCTLINE',ylabel='SALES',kind='bar')
plt.title("TOTAL PRODUCT SALES")
plt.show()'''
# ================= MACHINE LEARNING =================


# Features (inputs)
X = df[['QUANTITYORDERED', 'PRICEEACH', 'MONTH', 'YEAR']]

# Target (output)
y = df['SALES']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Show predictions
print("Predictions:", predictions[:5])

# Compare actual vs predicted
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': predictions
})

print(comparison.head())

# Error
error = mean_absolute_error(y_test, predictions)
print("MAE:", error)
r2 = r2_score(y_test, predictions)
print("R2 Score:", r2)
plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, color='red')  # perfect line
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()