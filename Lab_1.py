import pandas as pd
import matplotlib.pyplot as plt

#Đọc file CSV
df = pd.read_csv("ITA105_Lab_1.csv")

#Bai_1
print(df.shape)
print(df.describe())
print(df.isnull().sum())

#Bai_2
df["Category"].fillna(df["Category"].mode()[0], inplace=True)
df["Price"].fillna(df["Price"].mean(), inplace=True)
df["StockQuantity"].fillna(df["StockQuantity"].median(), inplace=True)

df_drop = df.dropna()
print(df_drop.shape)

#Bai_3
df.loc[df["StockQuantity"] < 0, "StockQuantity"] = df["StockQuantity"].median()
df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)]

#Bai_4
df["Price_Smoothed"] = df["Price"].rolling(5).mean()

plt.plot(df["Price"])
plt.plot(df["Price_Smoothed"])
plt.legend(["Gia goc", "Gia lam muot"])
plt.show()

#Bai_5
df["Category"] = df["Category"].str.lower()
df["Description"] = df["Description"].str.replace(r"[!?]+", "", regex=True)
df["Price_VND"] = df["Price"] * 24000

print(df.head())