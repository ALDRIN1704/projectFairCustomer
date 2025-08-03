import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

df = pd.read_csv("outputs/customer_risk_clusters.csv")

plt.figure(figsize=(8, 5))
sns.countplot(x='risk_level', data=df, order=['Low Risk', 'Medium Risk', 'High Risk'])
plt.title("Customer Count per Risk Group")
plt.xlabel("Risk Level")
plt.ylabel("Number of Customers")
plt.tight_layout()

os.makedirs("static/plots", exist_ok=True)
plt.savefig("static/plots/risk_cluster_distribution.png")
