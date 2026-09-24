# Correlation Heatmap

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# 1. Load Wine Dataset
wine = load_wine()

# 2. Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# 3. Calculate correlation matrix
corr_matrix = df.corr()

# 4. Print correlation matrix
print("--- Correlation Matrix ---")
print(corr_matrix)

# 5. Find strongest positive correlation
corr = corr_matrix.copy()

# Remove diagonal values
for i in range(len(corr)):
    corr.iloc[i, i] = -1

# Find feature pair
pair = corr.stack().idxmax()
value = corr.stack().max()

print("\n--- Strongest Positive Correlation ---")
print("Feature 1:", pair[0])
print("Feature 2:", pair[1])
print("Correlation:", value)

# 6. Plot heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap - Wine Dataset")
plt.tight_layout()
plt.show()