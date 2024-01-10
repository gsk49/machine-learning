import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file into a DataFrame
df = pd.read_csv('set3.csv')

# Group data by VID
grouped_data = df.groupby('VID')

# Create a scatter plot for each VID
for vid, group in grouped_data:
    plt.scatter(group['LON'], group['LAT'], label=f'VID {vid}')

# Add labels and legend
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trajectories with Color-coded VID')
plt.legend()

# Show the plot
plt.show()
