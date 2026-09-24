# Boxplots for all numerical attributes in Wine Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# 1. Load Wine Dataset
wine = load_wine()

# 2. Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# 3. Plot boxplots
plt.figure(figsize=(15, 7))

sns.boxplot(data=df)

plt.title("Boxplots of Wine Dataset Features")
plt.xlabel("Features")
plt.ylabel("Values")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()