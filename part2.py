import numpy as np
import matplotlib.pyplot as plt
import time
import RegressionTree as RT
from sklearn.model_selection import train_test_split

def generate_dataset():
    X = np.random.uniform(-3, 3, 100).reshape(-1, 1)
    y = 0.8 * np.sin(X - 1).ravel()
    return np.column_stack((X, y))

data = generate_dataset()
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

results = []
depth_limits = [None, 5, 10]
leaf_sizes = [None, 2, 4, 8]
type_limits = [None, 'height', 'leaf_size']


for type_limit in type_limits:
    for depth in depth_limits:
        for leaf_size in leaf_sizes:
            start_time = time.time()
            model = RT.RegressionTree(train_data, max_height=depth, min_leaf_size=leaf_size, limit_type=type_limit)
            build_time = time.time() - start_time
            
            test_X, test_y = test_data[:, :-1], test_data[:, -1]
            predictions = np.array([model.predict(x.reshape(1, -1)) for x in test_X])
            test_error = np.mean((test_y - predictions.ravel()) ** 2)
            
            results.append((type_limit, depth, leaf_size, build_time, test_error))

            sorted_indices = np.argsort(test_X[:, 0])
            sorted_X = test_X[sorted_indices]
            sorted_predictions = predictions.ravel()[sorted_indices]

            plt.figure(figsize=(8, 6))
            plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', label='Training Data', alpha=0.5)
            plt.scatter(test_X, test_y, color='red', label='Test Data')
            plt.plot(sorted_X, sorted_predictions, color='green', linewidth=2, label='Prediction')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(
                f'Regression Tree Approximation\nLimiter: {type_limit}, Depth: {depth}, Leaf Size: {leaf_size}\nBuild Time: {build_time:.4f}s, Test Error: {test_error:.4f}')
            plt.legend()
            plt.show()

print("Limiter  \t| Depth Limit   | Leaf Size     | Build Time (s) | Test Error")
for res in results:
    print(f"{res[0]}    \t| {res[1]}\t\t| {res[2]}\t\t|{res[3]:.4f}\t\t | {res[4]:.4f}")

sorted_indices = np.argsort(test_X[:, 0])
sorted_X = test_X[sorted_indices]
sorted_y = test_y[sorted_indices]
predictions = np.array([RT.RegressionTree.predict(model, x) for x in test_X])
sorted_predictions = predictions[sorted_indices]


plt.figure(figsize=(8, 6))
plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', label='Training Data', alpha=0.5)
plt.scatter(test_X, test_y, color='red', label='Test Data')
plt.plot(sorted_X, sorted_predictions, color='green', linewidth=2, label='Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression Tree Approximation')
plt.legend()
#plt.show()
