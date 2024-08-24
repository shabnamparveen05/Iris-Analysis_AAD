from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Convert the dataset to a DataFrame for better readability
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# Display the first five rows
print("First five rows of the dataset:")
print(iris_df.head())

# Display the shape of the dataset
print("\nShape of the dataset:", iris_df.shape)

# Summary statistics
print("\nSummary statistics for each feature:")
print(iris_df.describe())
