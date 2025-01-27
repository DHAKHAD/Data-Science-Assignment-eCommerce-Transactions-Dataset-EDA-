# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score

# Loading the Datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Task 1: Exploratory Data Analysis (EDA)
## Combine Datasets
merged_data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# Checking basic details of the dataset
print(merged_data.info())

# Visualizing Sales by Region
sales_by_region = merged_data.groupby("Region")["TotalValue"].sum().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x="Region", y="TotalValue", data=sales_by_region)
plt.title("Total Sales by Region")
plt.show()

# Top 5 Customers by Total Purchase
top_customers = merged_data.groupby("CustomerID")["TotalValue"].sum().sort_values(ascending=False).head(5)
print("Top 5 Customers by Total Purchase:")
print(top_customers)

# Task 2: Lookalike Model
## Creating a User-Item Matrix
user_item_matrix = merged_data.pivot_table(index="CustomerID", columns="ProductID", values="TotalValue", fill_value=0)

# Computing Similarity
similarity_matrix = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# Generating Lookalike Recommendations
lookalikes = {}
for customer in similarity_df.index[:20]:
    similar_customers = similarity_df[customer].sort_values(ascending=False).iloc[1:4]
    lookalikes[customer] = list(zip(similar_customers.index, similar_customers.values))

# Saving Lookalike Recommendations to CSV
lookalike_df = pd.DataFrame.from_dict(lookalikes, orient="index")
lookalike_df.to_csv("Lookalike.csv")

# Task 3: Customer Segmentation / Clustering
## Aggregating Transaction Data
customer_summary = merged_data.groupby("CustomerID").agg({
    "TotalValue": "sum",
    "Quantity": "sum"
}).reset_index()

# Scaling the Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_summary.drop("CustomerID", axis=1))

# Applying KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
customer_summary["Cluster"] = kmeans.fit_predict(scaled_data)

# Calculating Davies-Bouldin Index
db_index = davies_bouldin_score(scaled_data, customer_summary["Cluster"])
print(f"Davies-Bouldin Index: {db_index}")

# Visualizing Clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=customer_summary["Cluster"], palette="viridis")
plt.title("Customer Clusters")
plt.xlabel("Scaled Total Value")
plt.ylabel("Scaled Quantity")
plt.show()

# Saving Final Segmentation Results
customer_summary.to_csv("FirstName_LastName_Clustering.csv", index=False)