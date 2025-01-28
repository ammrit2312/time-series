import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import ruptures as rpt

# Read the data
PATH = './store-data'
data = pd.read_csv(os.path.join(PATH, 'oil.csv'), parse_dates=['date'], usecols=['date', 'dcoilwtico'])
data.set_index("date", inplace=True)
data = data.dropna(subset=['dcoilwtico'])
data = data.sort_index()

# Calculate the moving average of the raw data (using a window size of 28 days)
window_size = 28  # You can adjust this value based on your needs
data['moving_avg'] = data['dcoilwtico'].rolling(window=window_size).mean()

# Change point detection on the moving average values
moving_avg_data = data['moving_avg'].dropna().values  # Remove NaN values from moving average

# Use ruptures to detect change points on the moving average
algo = rpt.KernelCPD(kernel="linear", min_size=28)  #cosine, rbf, linear
algo.fit(moving_avg_data)
# change_points = algo.predict(pen=1)
change_points = algo.predict(n_bkps=10)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot the raw oil prices (blue line)
plt.plot(data.index, data['dcoilwtico'], color='blue', label='Raw Oil Price')

# Plot the moving average (orange line)
plt.plot(data.index, data['moving_avg'], color='orange', label='Moving Average')

# Plot the change points in the moving average (green vertical dashed lines)
for i, cp in enumerate(change_points):
    # We adjust the x-coordinate to match the corresponding date in the data index
    if i == 0:
        plt.axvline(x=data.index[cp], color='green', linestyle='--', label='Change Point')
    else:
        plt.axvline(x=data.index[cp], color='green', linestyle='--')

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Oil Price (WTI)")
plt.title(f"Oil Price Change Point Detection on Moving Average (Window Size: {window_size})")

# Format x-axis ticks to show dates in "Month Year" format
plt.xticks(data.index[::len(data)//10], data.index[::len(data)//10].strftime('%b %Y'), rotation=45)

# Add a legend
plt.legend(loc='upper left')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()