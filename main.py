import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

PATH = './store-data'

data = pd.read_csv(os.path.join(PATH, 'oil.csv'), parse_dates=['date'], usecols=['date', 'dcoilwtico'])
data.set_index("date", inplace=True)
data = data.dropna(subset=['dcoilwtico'])
data = data.sort_index()

print(data.head)


oilData = data["dcoilwtico"].values.reshape(-1, 1)

print(oilData)

import ruptures as rpt

# algo = rpt.Pelt(model = 'l2', min_size=28)
# algo.fit(oilData)
# result = algo.predict(pen=1)

# algo = rpt.Binseg(model="l2", min_size=28)
# algo.fit(oilData)
# result = algo.predict(n_bkps=10)

algo = rpt.Window(model="l2", width=28)
algo.fit(oilData)
result = algo.predict(n_bkps=5)

finalResult = []

# print([data.index[cp].strftime('%b %d, %Y') for cp in result])



# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(data.index, oilData, color='blue', label='Oil Price')

# Initialize variables to calculate means and variations for segments
start_idx = 0

for i, cp in enumerate(result):
    # Calculate the segment mean
    segment_mean = data["dcoilwtico"].iloc[start_idx:cp].mean()
    
    # Calculate the segment standard deviation (variation)
    segment_std = data["dcoilwtico"].iloc[start_idx:cp].std()

    # Plot the mean line for the segment
    plt.hlines(segment_mean, xmin=data.index[start_idx], xmax=data.index[cp-1], 
               colors='green', linestyles='-', linewidth=2, label='Mean' if i == 0 else None)

    # Plot the shaded area for variation (1 standard deviation)
    plt.fill_between(data.index[start_idx:cp], 
                     segment_mean - segment_std, 
                     segment_mean + segment_std, 
                     color='orange', alpha=0.3, label='Variation' if i == 0 else None)

    # Add a vertical line for the change point
    print(cp)
    print(finalResult, result[-1])
    if cp != result[-1]:  # Avoid marking the last point as a change point
        plt.axvline(x=data.index[cp], color='red', linestyle='--', label='Change Point' if i == 0 else None)
        finalResult.append(data.index[cp].strftime('%b %d, %Y') )

        # Display the change point date as a label
        change_point_date = data.index[cp].strftime('%b %d, %Y')  # Formatting the date as 'Month Day, Year'
        plt.text(data.index[cp], plt.ylim()[1] * 0.95, f'{change_point_date}', 
                 color='green', ha='center', va='top', fontsize=9)

    # Update the start index for the next segment
    start_idx = cp
print(finalResult)

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Oil Price (WTI)")
plt.title("Oil Price Change Point Detection with Segment Means and Variation")

# Format x-axis ticks to show dates in "Month Year" format
plt.xticks(data.index[::len(data)//10], data.index[::len(data)//10].strftime('%b %Y'), rotation=45)

# Add a legend
plt.legend(loc='upper left')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()