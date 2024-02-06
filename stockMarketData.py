import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# PART ONE -> GETTING DATA

# Get data & put it in a dataframe
df = pd.read_csv('stock_data.csv', usecols=['date', 'close', 'Name'])

print(df)

# PART TWO -> SORT & IDENTIFY UNIQUE NAMES 

unique_names = sorted(df['Name'].unique())
num_unique_names = len(unique_names)

print(f"\n Number of Unique Names: {num_unique_names}")
print(f"First 5 Names: {unique_names[:5]}")
print(f"Last 5 Names: {unique_names[-5:]} \n")


# PART THREE -> MIN REMOVE AFTER: 1st July 2014 or MAX BEFORE: 30th June 2017.

# Turn 'date' column to datetime format

df['date'] = pd.to_datetime(df['date'])  

# Find names to remove based on the dates given
names_to_remove = df.groupby('Name')['date'].agg(['min', 'max'])
names_to_remove = names_to_remove[
    (names_to_remove['min'] > '2014-07-01') | (names_to_remove['max'] < '2017-06-30')
].index

# Remove these names from the dataframe
df_filtered = df[~df['Name'].isin(names_to_remove)]

# Show the name that were removed
removed_names_count = len(names_to_remove)

print(f"Removed names: {list(names_to_remove)}")

# Show the number of names left ( those that meet critria )
remaining_names_count = df_filtered['Name'].nunique()

print(f"Number of names left: {remaining_names_count}")


# PART FOUR -> 4) FIND SET OF DATES COMMON TO REMAINING NAMES + REMOVE DATES BEFORE 1st July 2014 & AFTER 30th June 2017

# FIND COMMON DATES
common_dates = set(df_filtered.groupby('Name')['date'].agg(lambda x: set(x)).agg(lambda x: set.intersection(*x)))


# Remove dates that arent in range given  
df_filtered_dates = df_filtered[(df_filtered['date'] >= '2014-07-01') & (df_filtered['date'] <= '2017-06-30')]

# How many dates are left ? 
remaining_dates_count = df_filtered_dates['date'].nunique()
print(f"Number of dates left: {remaining_dates_count}")

# What are the first and last 5 dates ?
first_5_dates = df_filtered_dates['date'].sort_values().unique()[:5]
print(f"First 5 dates: {first_5_dates}")


last_5_dates = df_filtered_dates['date'].sort_values().unique()[-5:]
print(f"Last 5 dates: {last_5_dates}")


# PART FIVE -> dataframe: column == names from step (3). Rows == dates from step (4). “close” values for name and date. 

df_pivoted = df_filtered_dates.pivot(index='date', columns='Name', values='close')

# SHOW  df_pivoted
print(df_pivoted)


# PART SIX -> return(name, date) = (close(name, date) - close(name, previous date)) / close(name, previous date) -->
#  NEW - OLD / OLD --> PERCENT CHANGE RELATIVE TO PREVIOUS VLAUE

# pct_change() calculates change form one period to the next.
# dropna() removes the first row; in this case it would only contain Null values
returns_df = df_pivoted.pct_change().dropna()

# Show result 
print(returns_df)


# PART SEVEN --> PCA : PC RANKED BY THE EIGENVALUE SIZE

# Initialising PCA w/ the number of columns
PrincCompAnalysis = PCA(n_components=len(returns_df.columns))

# Fit PCA Model
PrincCompAnalysis.fit(returns_df)

# extract principal components
principalComponents = PrincCompAnalysis.components_

# find eigenvalues & sort them ( descending order ) largest effect to smallest effect
eigenvaluesSorted = sorted(range(len(PrincCompAnalysis.explained_variance_)), key=lambda k: PrincCompAnalysis.explained_variance_[k], 
                           reverse=True)

# Showing top 5 Principal Components

print("Top 5 Principal Components:")

for i in range(5):
    pc_index = eigenvaluesSorted[i]
    eigenvalue = PrincCompAnalysis.explained_variance_[pc_index]
    print(f"Principal Component # {i+1}: Eigenvalue = {eigenvalue:.4f} \n")
    print(f"{principalComponents[pc_index]} \n")


# PART EIGHT --> " extract the explained variance ratios. "

# Explained variance ratios 
explained_variance_ratios = PrincCompAnalysis.explained_variance_ratio_

#  Percent of variance explained by the first principal component
percentVarianceFirstPC = explained_variance_ratios[0] * 100

# Display the percentage of variance explained by the first principal component
print(f"What percentage of variance is explained by the first principal component? {percentVarianceFirstPC:.2f}% \n")

# First 20 explained variance ratios

plt.figure(figsize=(10, 6))

plt.plot(range(1, 21), explained_variance_ratios[:20], marker='o', linestyle='-', color='b')

plt.title('Explained Variance Ratios for PC')
plt.xlabel('PC ( Principle Components )')
plt.ylabel('Explained Variance Ratio')

plt.xticks(range(1, 21))

plt.grid(True)

# Finding the elbow point
elbow_index = 4 
elbow_point = (elbow_index + 1, explained_variance_ratios[elbow_index])
plt.annotate(f'Elbow\nPC {elbow_point[0]}', elbow_point, textcoords="offset points", xytext=(-15,-10), ha='center', fontsize=9, 
             color='r', weight='bold')

# Show the plot
plt.show()

# PART NINE -> Calculate cumulative variance ratios using numpy. Plot cumulative variance ratios
#  (x axis = principal component, y axis = cumulative variance ratio).

# Mark on your plot the principal component for which the cumulative variance ratio is greater than or equal to 95%.

# Cumulative Variance Ratios with numpy cumsum function
cumulativeVarianceRatios = np.cumsum(explained_variance_ratios)

# Cumulative variance ratio is greater than or equal to 95%.
greaterThanOrEqualto95 = np.argmax(cumulativeVarianceRatios >= 0.95)

# Plot principle component for cumulative variance ratio
plt.figure(figsize=(10, 6))

plt.plot(range(1, len(cumulativeVarianceRatios) + 1), cumulativeVarianceRatios, marker='o', linestyle='-', color='b')
plt.axvline(x=greaterThanOrEqualto95 + 1, color='r', linestyle='--', label='95% Threshold')

# Find cumulative variance is greater than or equal to 95%
plt.scatter(greaterThanOrEqualto95 + 1, cumulativeVarianceRatios[greaterThanOrEqualto95], color='r', s=100, label=f'PC {greaterThanOrEqualto95 + 1}')

plt.title('Cumulative Variance Ratios')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Variance Ratio')

plt.xticks(range(1, len(cumulativeVarianceRatios) + 1, 24))
plt.grid(True)
plt.legend()

plt.show()


# PART TEN -> Normalise your dataframe from step (6) 
# columns have zero mean and unit variance. 
# Repeat steps (7) - (9) for this new dataframe.

# Normalise dataframe
scaler = StandardScaler()
returnsNormalized_df = scaler.fit_transform(returns_df)

# PCA --> normalised dataframe
PCANormalized = PCA(n_components=len(returns_df.columns))

PCANormalized.fit(returnsNormalized_df)

# Explained Variance Ratios
explainedVarianceRatiosNormalized = PCANormalized.explained_variance_ratio_

# Cumulative Variance Ratios 
cumulativeVarianceRatiosNormalized = np.cumsum(explainedVarianceRatiosNormalized)

# Cumulative variance greater than / equal to 95% 
greaterThanOrEqualto95Normalized = np.argmax(cumulativeVarianceRatiosNormalized >= 0.95)

# Plotting the Cumulative Variance Ratios; but now for the normalized dataframe
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulativeVarianceRatiosNormalized) + 1), cumulativeVarianceRatiosNormalized, marker='o', linestyle='-', color='b')
plt.axvline(x=greaterThanOrEqualto95Normalized + 1, color='r', linestyle='--', label='95% Threshold')

# Mark cumulative variance >= 95%
plt.scatter(greaterThanOrEqualto95Normalized + 1, cumulativeVarianceRatiosNormalized[greaterThanOrEqualto95Normalized], 
            color='r', s=100, label=f'PC {greaterThanOrEqualto95Normalized + 1}')

plt.title('Cumulative Variance Ratios (Normalized)')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Variance Ratio')
plt.xticks(range(1, len(cumulativeVarianceRatiosNormalized) + 1, 24))
plt.grid(True)
plt.legend()

plt.show()
