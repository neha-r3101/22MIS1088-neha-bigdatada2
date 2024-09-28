import pandas as pd
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_excel('/Users/neharavindran/Downloads/Online Retail.xlsx', engine='openpyxl')

# Data Preprocessing
df = df.dropna(subset=['CustomerID'])

# Aggregating customer data
customer_data = df.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'UnitPrice': 'mean',
    'InvoiceNo': 'nunique'
}).reset_index()

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
customer_data_scaled = scaler.fit_transform(customer_data.drop('CustomerID', axis=1))

# Initialize and train SOM
som = MiniSom(10, 10, customer_data_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(customer_data_scaled)
som.train_random(customer_data_scaled, 1000)

# Visualize the results
wmap = {}
for x, t in zip(customer_data_scaled, range(len(customer_data_scaled))):
    w = som.winner(x)
    wmap[w] = wmap.get(w, []) + [t]

plt.figure(figsize=(10, 10))
for i in range(som.get_weights().shape[0]):
    for j in range(som.get_weights().shape[1]):
        plt.text(j, i, str(len(wmap.get((i, j), []))),
                 ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5, lw=0))

plt.imshow(np.zeros((som.get_weights().shape[0], som.get_weights().shape[1])), cmap='bone_r')
plt.colorbar()
plt.title('SOM Clustering Visualization')
plt.show()

# Save the clustered customer data
customer_data['Cluster'] = [som.winner(x) for x in customer_data_scaled]
customer_data.to_csv('clustered_customers.csv', index=False)

print("Clustering complete. Data saved to clustered_customers.csv.")

