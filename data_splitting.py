from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=42)

# Print the number of samples in both the training and testing sets
print("Number of samples in the training set:", len(X_train))
print("Number of samples in the testing set:", len(X_test))
