import numpy as np
from sklearn.model_selection import train_test_split
import RegressionTree as RT
import matplotlib.pyplot as plt
import pandas as pd

# Function to generate samples
def generate_system_samples(n_samples=500):
    x1 = np.random.uniform(-5, 5, n_samples)
    x2 = np.random.uniform(-5, 5, n_samples)
    x1_next = 0.9 * x1 - 0.2 * x2
    x2_next = 0.2 * x1 + 0.9 * x2
    current_states = np.column_stack((x1, x2))
    next_states = np.column_stack((x1_next, x2_next))
    return current_states, next_states


# Predict using two trees
def predict_multi_output(tree1, tree2, X):
    pred1 = np.array([tree1.predict(x) for x in X])
    pred2 = np.array([tree2.predict(x) for x in X])
    return np.column_stack((pred1, pred2))


X, y = generate_system_samples()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare trees
train_data_tree1 = np.column_stack((X_train, y_train[:, 0]))
train_data_tree2 = np.column_stack((X_train, y_train[:, 1]))

# Train trees
tree1 = RT.RegressionTree(train_data_tree1, limit_type=None)
tree2 = RT.RegressionTree(train_data_tree2, limit_type=None)

# actual and predicted
initial_state = np.array([0.5, 1.5]).reshape(1, -1)
predicted_trajectory = [initial_state.flatten()]
actual_trajectory = [initial_state.flatten()]
comparison_table = []

current_state = initial_state.copy()
for step in range(20):
    pred_next = predict_multi_output(tree1, tree2, current_state).flatten()
    true_next = np.dot(np.array([[0.9, -0.2], [0.2, 0.9]]), current_state.flatten())

    predicted_trajectory.append(pred_next)
    actual_trajectory.append(true_next)
    comparison_table.append([step, *true_next, *pred_next])
    current_state = true_next.reshape(1, -1)

predicted_trajectory = np.array(predicted_trajectory)
actual_trajectory = np.array(actual_trajectory)

# Plot for Phase Plot: Actual vs Predicted Trajectory
plt.figure(figsize=(8, 6))
plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], 'bo-', label='Actual Trajectory')
plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'ro--', label='Predicted Trajectory')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Phase Plot: Actual vs Predicted Trajectory')
plt.legend()
plt.grid(True)
plt.show()

# Plot for State Evolution Over Time
plt.figure(figsize=(10, 5))
plt.plot(predicted_trajectory[:, 0], label="Predicted x1", linestyle='--', marker='o')
plt.plot(predicted_trajectory[:, 1], label="Predicted x2", linestyle='--', marker='o')
plt.plot(actual_trajectory[:, 0], label="Actual x1", linestyle='-', marker='x')
plt.plot(actual_trajectory[:, 1], label="Actual x2", linestyle='-', marker='x')
plt.xlabel("Time Step")
plt.ylabel("State Value")
plt.title("State Evolution Over Time")
plt.legend()
plt.grid(True)
plt.show()


