import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pension = "C:/Users/hp/Downloads/analysis.csv" 
df = pd.read_csv(pension)
print("Dataset Overview:")
print(df.info())
df.fillna(0, inplace=True)

print("\n")
print("Summary Statistics:")
print(df.describe())


df.columns = df.columns.str.lower()


plt.figure(figsize=(12, 6))
state_beneficiaries = df.groupby('state_name')['total_beneficiaries'].sum().sort_values(ascending=False)
state_beneficiaries.plot(kind='bar', color='skyblue')
plt.title("Total Beneficiaries per State")
plt.xlabel("State")
plt.ylabel("Total Beneficiaries")
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(10, 5))
categories = ["total_sc", "total_st", "total_gen", "total_obc"]
df[categories].sum().plot(kind='bar', color=['red', 'green', 'blue', 'purple'])
plt.title("Distribution of Beneficiaries by Category")
plt.xlabel("Category")
plt.ylabel("Total Count")
plt.show()


numeric_df = df.select_dtypes(include=['number']) 


plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()



if 'mnth' in df.columns:
    monthly_trend = df.groupby('mnth')['total_beneficiaries'].sum()
    plt.figure(figsize=(8, 5))
    monthly_trend.plot(marker='o', linestyle='-', color='orange')
    plt.title("Trend of Total Beneficiaries Over Months")
    plt.xlabel("Month")
    plt.ylabel("Total Beneficiaries")
    plt.grid(True)
    plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=df[['total_beneficiaries', 'total_aadhar', 'total_mobileno']], palette="coolwarm")
plt.title("Outlier Detection in Key Metrics")
plt.show()
