# Description: This code uses a simple KMeans clustering algorithm to group Titanic passengers. Clustering is unsupervised learning, which means we don't tell it who survived or not — it just groups based on similarities.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Hi, I'm Narin. For this project, I wanted to try out unsupervised learning using the Titanic dataset.
# First, I load the Excel file with all the data.
df = pd.read_excel(r"C:\Users\nmane\Documents\Machine Learning Projects\Titanic Disaster Project\titanicdisaster_dataset.xlsx")

# I picked a few columns that seemed useful for clustering: class, age, fare, and sibsp (siblings/spouses aboard).
# I also kept names so I can show them in the results.
df_clean = df[['name', 'pclass', 'age', 'fare', 'sibsp']].dropna()  # Drop rows with missing values

# KMeans works better when numbers are on the same scale, so I scaled them here.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean[['pclass', 'age', 'fare', 'sibsp']])

# I used 3 clusters to see how it groups the passengers (like rich, middle, poor maybe).
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(scaled_data)

# Add the cluster number to the original data so we can see which passenger is in which group
df_clean['Cluster'] = labels

# Plotting age vs. fare to visualize the clusters. I kept it basic.
colors = ['red', 'green', 'blue']
for i in range(3):
    group = df_clean[df_clean['Cluster'] == i]
    plt.scatter(group['age'], group['fare'], color=colors[i], label=f'Cluster {i}')

plt.title('Titanic Passenger Clusters')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend()
plt.show()

# Print cluster centers just to check what the average values are for each group.
print("Cluster centers:")
print(kmeans.cluster_centers_)

# Show some sample passengers and their cluster numbers
print("\nSample passengers with clusters:")
print(df_clean[['name', 'pclass', 'age', 'fare', 'sibsp', 'Cluster']].head(10))