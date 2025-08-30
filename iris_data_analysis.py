# ========================================================
# Assignment: Data Loading, Analysis, and Visualization
# Student: [Whitney Wairimu]
# ========================================================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ------------------------------
# Task 1: Load and Explore the Dataset
# ------------------------------

try:
    # Load iris dataset from sklearn
    iris = load_iris(as_frame=True)
    data = iris.frame  # Convert to DataFrame
    
    print("✅ Dataset loaded successfully!\n")
except Exception as e:
    print("❌ Error loading dataset:", e)

# Display first few rows
print("📊 First 5 rows of the dataset:")
print(data.head(), "\n")

# Explore structure
print("ℹ️ Info about the dataset:")
print(data.info(), "\n")

# Check missing values
print("🔎 Missing values in each column:")
print(data.isnull().sum(), "\n")

# (Iris dataset has no missing values, but if there were, we could drop or fill them)
data = data.dropna()  # Cleaning step

# ------------------------------
# Task 2: Basic Data Analysis
# ------------------------------

# Basic statistics
print("📈 Basic Statistics:")
print(data.describe(), "\n")

# Group by species and calculate mean
grouped = data.groupby("target").mean()
print("📊 Mean values by species (target):")
print(grouped, "\n")

# Observations
print("📝 Observations:")
print("- Sepal length and width differ significantly across species.")
print("- Petal length and width are good features for distinguishing species.\n")

# ------------------------------
# Task 3: Data Visualization
# ------------------------------

# Style settings
sns.set(style="whitegrid")

# 1. Line chart (trend over index, just to demonstrate time-series like trend)
plt.plot(data.index, data["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart: Average petal length per species
sns.barplot(x="target", y="petal length (cm)", data=data, estimator="mean", palette="Set2")
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species (0=setosa, 1=versicolor, 2=virginica)")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram: Sepal length distribution
plt.hist(data["sepal length (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot: Sepal length vs Petal length
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="target", palette="deep", data=data)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# ------------------------------
# Findings / Observations
# ------------------------------
print("📌 Final Findings:")
print("1. Iris-setosa species has much smaller petal lengths compared to others.")
print("2. Iris-virginica species has the largest petal dimensions overall.")
print("3. Sepal measurements overlap across species, but petal dimensions separate them well.")
print("4. Scatter plot shows clear clusters → useful for classification problems.")
