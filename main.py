# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# PATH = './store-data'

# data = pd.read_csv(os.path.join(PATH, 'oil.csv'), parse_dates=['date'], usecols=['date', 'dcoilwtico'])
# data.set_index("date", inplace=True)
# data = data.dropna(subset=['dcoilwtico'])
# data = data.sort_index()

# print(data.head)


# oilData = data["dcoilwtico"].values.reshape(-1, 1)

# print(oilData)

# import ruptures as rpt

# # algo = rpt.Pelt(model = 'l2', min_size=28)
# # algo.fit(oilData)
# # result = algo.predict(pen=1)

# # algo = rpt.Binseg(model="l2", min_size=28)
# # algo.fit(oilData)
# # result = algo.predict(n_bkps=10)

# algo = rpt.Window(model="l2", width=28)
# algo.fit(oilData)
# result = algo.predict(n_bkps=5)

# finalResult = []

# # print([data.index[cp].strftime('%b %d, %Y') for cp in result])



# # Plot the time series
# plt.figure(figsize=(12, 6))
# plt.plot(data.index, oilData, color='blue', label='Oil Price')

# # Initialize variables to calculate means and variations for segments
# start_idx = 0

# for i, cp in enumerate(result):
#     # Calculate the segment mean
#     segment_mean = data["dcoilwtico"].iloc[start_idx:cp].mean()
    
#     # Calculate the segment standard deviation (variation)
#     segment_std = data["dcoilwtico"].iloc[start_idx:cp].std()

#     # Plot the mean line for the segment
#     plt.hlines(segment_mean, xmin=data.index[start_idx], xmax=data.index[cp-1], 
#                colors='green', linestyles='-', linewidth=2, label='Mean' if i == 0 else None)

#     # Plot the shaded area for variation (1 standard deviation)
#     plt.fill_between(data.index[start_idx:cp], 
#                      segment_mean - segment_std, 
#                      segment_mean + segment_std, 
#                      color='orange', alpha=0.3, label='Variation' if i == 0 else None)

#     # Add a vertical line for the change point
#     print(cp)
#     print(finalResult, result[-1])
#     if cp != result[-1]:  # Avoid marking the last point as a change point
#         plt.axvline(x=data.index[cp], color='red', linestyle='--', label='Change Point' if i == 0 else None)
#         finalResult.append(data.index[cp].strftime('%b %d, %Y') )

#         # Display the change point date as a label
#         change_point_date = data.index[cp].strftime('%b %d, %Y')  # Formatting the date as 'Month Day, Year'
#         plt.text(data.index[cp], plt.ylim()[1] * 0.95, f'{change_point_date}', 
#                  color='green', ha='center', va='top', fontsize=9)

#     # Update the start index for the next segment
#     start_idx = cp
# print(finalResult)

# # Add labels and title
# plt.xlabel("Date")
# plt.ylabel("Oil Price (WTI)")
# plt.title("Oil Price Change Point Detection with Segment Means and Variation")

# # Format x-axis ticks to show dates in "Month Year" format
# plt.xticks(data.index[::len(data)//10], data.index[::len(data)//10].strftime('%b %Y'), rotation=45)

# # Add a legend
# plt.legend(loc='upper left')

# # Adjust layout and show the plot
# plt.tight_layout()
# plt.show()







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

# Define function to compute means over segments
def calculate_segment_means(data, change_points):
    means = []
    start_idx = 0
    for cp in change_points:
        # Check if it's the last change point to avoid going out of bounds
        if cp == change_points[-1]:
            segment_mean = data["dcoilwtico"].iloc[start_idx:].mean()
        else:
            segment_mean = data["dcoilwtico"].iloc[start_idx:cp].mean()
        means.append(segment_mean)
        start_idx = cp
    return np.array(means)

# Change point detection on raw data
oilData = data["dcoilwtico"].values

# Use ruptures to detect change points on the raw data
algo = rpt.Pelt(model="l2", min_size=28)
algo.fit(oilData)
change_points = algo.predict(pen=1)

# Calculate segment means for each detected segment
segment_means = calculate_segment_means(data, change_points)

# Apply ruptures to the segment means
algo_means = rpt.Pelt(model="l2", min_size=1)  # Apply change point detection on mean values
algo_means.fit(segment_means)
result_means = algo_means.predict(pen=1)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot the original oil prices (blue line)
plt.plot(data.index, oilData, color='blue', label='Oil Price')

# Plot the change points in the original data
for i, cp in enumerate(change_points):
    if i == 0:
        plt.axvline(x=data.index[cp], color='red', linestyle='--', label='Change Point')
    else:
        plt.axvline(x=data.index[cp], color='red', linestyle='--')

# Plot the change points in the segment means
for i, cp in enumerate(result_means[:-1]):  # Exclude the last change point
    plt.axvline(x=data.index[change_points[cp]], color='green', linestyle='--', label='Mean Change Point' if i == 0 else None)

# Labeling the segments
start_idx = 0
for i, cp in enumerate(change_points):
    # Check if it's the last change point to avoid out-of-bounds error
    if cp == change_points[-1]:
        segment_mean = np.mean(oilData[start_idx:])  # Mean of the last segment
    else:
        segment_mean = np.mean(oilData[start_idx:cp])  # Mean of each segment
    plt.hlines(segment_mean, xmin=data.index[start_idx], xmax=data.index[cp-1], colors='green', linestyle='-', linewidth=2)
    start_idx = cp

# Labels and title
plt.xlabel("Date")
plt.ylabel("Oil Price (WTI)")
plt.title("Oil Price Change Point Detection on Mean Values")

# Format x-axis ticks
plt.xticks(data.index[::len(data)//10], data.index[::len(data)//10].strftime('%b %Y'), rotation=45)

# Add a legend
plt.legend(loc='upper left')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()