import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Loat Data
df = pd.read_csv("StockPriceDataset.csv")
print("First 5 Rows:")
print(df.head())
print("\nColumns:")
print(df.columns)
print("\nMissing Values:")
print(df.isnull().sum())


#Data Cleaning
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values(by='Date')
df = df.reset_index(drop=True)
print("\nData After Cleaning:")
print(df.head())

# Obj 1: Company with high volume
company_volume = df.groupby('Ticker')['Volume'].sum().sort_values(ascending=False)
print("\nTotal Trading Volume by Company:")
print(company_volume)
plt.figure()
plt.bar(company_volume.index, company_volume.values)
plt.title("Company-wise Trading Volume")
plt.xlabel("Company")
plt.ylabel("Total Volume")
plt.xticks(rotation=45)
plt.show()

#Select top company
top_company = company_volume.idxmax()
print("\nTop Company:", top_company)
aapl_df = df[df['Ticker'] == top_company].copy()


# Obj 2: Stock trend of allp
plt.figure()
plt.plot(aapl_df['Date'], aapl_df['Close'])
plt.title(f"{top_company} Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()


# Obj 3: Monthly analysis of allp
aapl_df['Month'] = aapl_df['Date'].dt.month
monthly_avg = aapl_df.groupby('Month')['Close'].mean()
print("\nMonthly Average Price:")
print(monthly_avg)
plt.figure()
plt.bar(monthly_avg.index, monthly_avg.values)
plt.title("Monthly Average Price")
plt.xlabel("Month")
plt.ylabel("Price")
plt.show()

# Obj 4: Volatility
aapl_df['Volatility'] = aapl_df['High'] - aapl_df['Low']
print("\nVolatility Sample:")
print(aapl_df['Volatility'].head())
plt.figure()
plt.hist(aapl_df['Volatility'], bins=20)
plt.title("Volatility Distribution")
plt.show()


# Obj 5: Correlation
corr = aapl_df[['Close','Volume']].corr()
print("\nCorrelation:")
print(corr)
plt.figure()
sns.heatmap(corr, annot=True)
plt.title("Volume vs Price")
plt.show()

# Checking for outlier
plt.figure()
plt.boxplot(aapl_df['Close'])
plt.title("Outliers in Closing Price")
plt.ylabel("Price")
plt.show()
Q1 = aapl_df['Close'].quantile(0.25)
Q3 = aapl_df['Close'].quantile(0.75)
IQR = Q3 - Q1
outliers = aapl_df[(aapl_df['Close'] < Q1 - 1.5*IQR) | 
                   (aapl_df['Close'] > Q3 + 1.5*IQR)]
print("\nNumber of Outliers:", len(outliers))


# Obj 6: Moving avg
aapl_df['MA7'] = aapl_df['Close'].rolling(7).mean()
aapl_df['MA30'] = aapl_df['Close'].rolling(30).mean()
plt.figure()
plt.plot(aapl_df['Date'], aapl_df['Close'], label='Price')
plt.plot(aapl_df['Date'], aapl_df['MA7'], label='MA7')
plt.plot(aapl_df['Date'], aapl_df['MA30'], label='MA30')
plt.legend()
plt.title("Moving Average")
plt.show()

#STATISTICAL TESTS
#Normality Test
stat, p = stats.shapiro(aapl_df['Close'].sample(min(5000, len(aapl_df))))
print("\nShapiro Test p-value:", p)

# T-test
high_volume = aapl_df[aapl_df['Volume'] > aapl_df['Volume'].median()]['Close']
low_volume = aapl_df[aapl_df['Volume'] <= aapl_df['Volume'].median()]['Close']
t_stat, p_val = stats.ttest_ind(high_volume, low_volume)
print("\nT-Test Result:")
print("T-statistic:", t_stat)
print("P-value:", p_val)

if p_val < 0.05:
    print("Significant difference between high and low volume prices")
else:
    print("No significant difference")

#Model accuracy
X = aapl_df[['Open','High','Low','Volume']]
y = aapl_df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("\nModel Accuracy:", score)

