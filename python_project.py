import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'your_file.csv' with actual file path)
file_path = "C:/Users/hp/Downloads/analysis.csv"  # Update this with your actual file path
df = pd.read_csv(file_path)

# Display dataset information
print("Dataset Overview:")
print(df.info())

# Handling Missing Values (fill with 0 for numeric columns)
df.fillna(0, inplace=True)

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Convert column names to lowercase for easy access
df.columns = df.columns.str.lower()

### 1. Total Beneficiaries per State ###
plt.figure(figsize=(12, 6))
state_beneficiaries = df.groupby('state_name')['total_beneficiaries'].sum().sort_values(ascending=False)
state_beneficiaries.plot(kind='bar', color='skyblue')
plt.title("Total Beneficiaries per State")
plt.xlabel("State")
plt.ylabel("Total Beneficiaries")
plt.xticks(rotation=90)
plt.show()

### 2. Distribution of Beneficiaries by Category ###
plt.figure(figsize=(10, 5))
categories = ["total_sc", "total_st", "total_gen", "total_obc"]
df[categories].sum().plot(kind='bar', color=['red', 'green', 'blue', 'purple'])
plt.title("Distribution of Beneficiaries by Category")
plt.xlabel("Category")
plt.ylabel("Total Count")
plt.show()

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['number'])  # Excludes strings

# Generate correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


### 4. Trend Analysis: Monthly Beneficiaries ###
if 'mnth' in df.columns:
    monthly_trend = df.groupby('mnth')['total_beneficiaries'].sum()
    plt.figure(figsize=(8, 5))
    monthly_trend.plot(marker='o', linestyle='-', color='orange')
    plt.title("Trend of Total Beneficiaries Over Months")
    plt.xlabel("Month")
    plt.ylabel("Total Beneficiaries")
    plt.grid(True)
    plt.show()

### 5. Outlier Detection Using Boxplot ###
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[['total_beneficiaries', 'total_aadhar', 'total_mobileno']], palette="coolwarm")
plt.title("Outlier Detection in Key Metrics")
plt.show()
