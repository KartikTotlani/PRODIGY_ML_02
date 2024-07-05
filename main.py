import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('E:/prodigy_intern/mlt_02/SampleSuperstore.csv')

# Display the first few rows of the dataset
print(data.head())

# Feature Selection
# Assuming the dataset has columns 'CustomerID', 'TotalSpent', 'PurchaseFrequency', 'AverageOrderValue'
# Adjust feature names based on your actual dataset
features = data[['Sales', 'Quantity', 'Profit']]

# Data Standardization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Set the number of clusters
kmeans.fit(scaled_features)

# Add the cluster labels to the original data
data['Cluster'] = kmeans.labels_

# Display the first few rows of the dataset with the cluster labels
print(data.head())

# Evaluate Clusters
# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sales', y='Quantity', hue='Cluster', data=data, palette='viridis')
plt.title('Customer Segments Based on Purchase History')
plt.xlabel('Sales')
plt.ylabel('Quantity')
plt.legend(title='Cluster')
plt.show()

# Optional: Analyze the clusters
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_df = pd.DataFrame(cluster_centers, columns=['Sales', 'Quantity', 'Profit'])
print("Cluster Centers:\n", cluster_df)
