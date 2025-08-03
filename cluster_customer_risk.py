import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ========== Step 1: Load Dataset ==========
df = pd.read_csv("data/Accounts-Receivable.csv")

# ========== Step 2: Aggregate by customerID ==========
customer_data = df.groupby('customerID').agg({
    'invoiceNumber': 'count',
    'DaysLate': ['mean', 'max', 'sum'],
    'InvoiceAmount': 'mean',
    'Disputed': lambda x: (x == 'Yes').sum()
}).reset_index()

# Rename columns
customer_data.columns = [
    'customerID',
    'total_invoices',
    'avg_days_late',
    'max_days_late',
    'total_days_late',
    'avg_invoice_amount',
    'total_disputes'
]

# ========== Step 3: Scale the Data ==========
features = ['total_invoices', 'avg_days_late', 'max_days_late', 'total_days_late', 'avg_invoice_amount', 'total_disputes']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data[features])

# ========== Step 4: Apply KMeans Clustering ==========
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_data['risk_cluster'] = kmeans.fit_predict(X_scaled)

# ========== Step 5: Rename Risk Clusters for Readability ==========
# Optional: Based on cluster centers, we map 0/1/2 to Low, Medium, High Risk
cluster_map = {}

# Analyze average lateness per cluster
cluster_summary = customer_data.groupby('risk_cluster')[['avg_days_late']].mean().sort_values('avg_days_late')

# Assign labels
for idx, label in zip(cluster_summary.index, ['Low Risk', 'Medium Risk', 'High Risk']):
    cluster_map[idx] = label

customer_data['risk_level'] = customer_data['risk_cluster'].map(cluster_map)

# ========== Step 6: Save Results ==========
os.makedirs("outputs", exist_ok=True)
customer_data.to_csv("outputs/customer_risk_clusters.csv", index=False)
print("✅ Saved customer risk data to outputs/customer_risk_clusters.csv")

# ========== Step 7: Visualization ==========
plt.figure(figsize=(8, 5))
sns.countplot(x='risk_level', data=customer_data, order=['Low Risk', 'Medium Risk', 'High Risk'])
plt.title("Customer Count per Risk Group")
plt.xlabel("Risk Level")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/risk_cluster_distribution.png")
plt.show()

# ========== Step 8: Display Sample Output ==========
print("\n📌 Sample Output:")
print(customer_data[['customerID', 'total_invoices', 'avg_days_late', 'total_disputes', 'risk_level']].head())
