import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read CSV file into a DataFrame
df = pd.read_csv('newcsv.csv')

# Convert 'SEQUENCE_DTTM' to numerical values
df['SEQUENCE_DTTM'] = pd.to_datetime(df['SEQUENCE_DTTM']).astype('int64')

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Group data by VID
grouped_data = df.groupby('CLASS')

# Create a scatter plot for each VID in 3D
for cl, group in grouped_data:
    ax.scatter(group['LON'], group['LAT'],
               group['SEQUENCE_DTTM'])

# Add labels and legend
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('SEQUENCE_DTTM')
ax.set_title('set3 Predictions')
ax.legend()

# Show the plot
plt.show()
