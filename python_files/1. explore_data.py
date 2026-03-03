# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Explore the Midwest Survey dataset

# %%
from skrub.datasets import fetch_midwest_survey
import matplotlib.pyplot as plt

# Load the dataset
dataset = fetch_midwest_survey()
X = dataset.X
y = dataset.y

# %% [markdown]
# ## Question 1: How many examples are there in the dataset?

# %%
# Display the number of rows and columns
print(f"Rows: {X.shape[0]}, Columns: {X.shape[1]}")
X.head()

# %% [markdown]
# ## Question 2: What is the distribution of the target?

# %%
# Count how many respondents belong to each region
print(y.value_counts())

# Visualize the target distribution with a bar plot
y.value_counts().plot(kind='barh', color='skyblue')
plt.title("Distribution of Census Regions")
plt.xlabel("Count")
plt.show()

# %% [markdown]
# ## Question 3: What are the features that can be used to predict the target?

# %%
# List all column names
print(X.columns.tolist())

# Show data types for each column
print(X.dtypes)

# %%
from skrub import TableReport
TableReport(X)

# %% [markdown]
# ## Question 4: Are there any missing values in the dataset?

# %%
# Check for NaN missing values
print(X.isna().sum())

# Look at unique values for specific columns to find implicit missing values
print("Unique Household_Income:", X["Household_Income"].unique())
print("Unique Education:", X["Education"].unique())

# %% [markdown]
# ## Question 5: What is the most common answer to "How much do you personally identify as a Midwesterner"?

# %%
col_id = "How_much_do_you_personally_identify_as_a_Midwesterner"

# Display the value counts
identify_counts = X[col_id].value_counts