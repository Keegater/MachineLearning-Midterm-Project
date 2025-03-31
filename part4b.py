import numpy as np
import matplotlib.pyplot as plt
import RegressionTree as RT
import pandas as pd
from sklearn.model_selection import train_test_split


# One-step transition of the given program
def one_step(x, z):
    if x > 1:
        x_next = 0
    else:
        x_next = x + 0.2
    z_next = z + x_next
    return x_next, z_next


# Generate training data: (x, z) -> (x', z')
def generate_program_samples(n_samples=1000):
    x_vals = np.random.uniform(-3, 3, n_samples)
    z_vals = np.random.uniform(0, 15, n_samples)

    X = np.column_stack((x_vals, z_vals))
    y = np.array([one_step(x, z) for x, z in X])

    return X, y


# Predict both x' and z' using two regression trees
def predict_next_state(tree_x, tree_z, X):
    pred_x = np.array([tree_x.predict(x) for x in X])
    pred_z = np.array([tree_z.predict(x) for x in X])
    return np.column_stack((pred_x, pred_z))


# === Generate training data ===
X, y = generate_program_samples()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train trees for x' and z'
train_data_x = np.column_stack((X_train, y_train[:, 0]))
train_data_z = np.column_stack((X_train, y_train[:, 1]))

tree_x = RT.RegressionTree(train_data_x, limit_type=None)
tree_z = RT.RegressionTree(train_data_z, limit_type=None)

# === Simulate actual and predicted trajectory from initial state ===
initial_state = np.array([2.0, 0.0]).reshape(1, -1)
actual_trajectory = [initial_state.flatten()]
predicted_trajectory = [initial_state.flatten()]
comparison_table = []

current_state = initial_state.copy()
for step in range(20):
    # True system step
    true_x, true_z = one_step(current_state[0, 0], current_state[0, 1])
    true_next = np.array([true_x, true_z])
    actual_trajectory.append(true_next)

    # Model prediction
    pred_next = predict_next_state(tree_x, tree_z, current_state).flatten()
    predicted_trajectory.append(pred_next)

    comparison_table.append(
        [step, current_state[0, 0], current_state[0, 1], true_x, true_z, pred_next[0], pred_next[1]])

    # Move to next actual state
    current_state = true_next.reshape(1, -1)

actual_trajectory = np.array(actual_trajectory)
predicted_trajectory = np.array(predicted_trajectory)

# === Plot Phase Space Trajectory (x vs z) ===
plt.figure(figsize=(8, 6))
plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], 'bo-', label='Actual Trajectory')
plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'ro--', label='Predicted Trajectory')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Program State Trajectory (x, z)')
plt.legend()
plt.grid(True)
plt.show()

# === Plot x and z separately over time ===
plt.figure(figsize=(10, 5))
plt.plot([a[0] for a in actual_trajectory], label="Actual x", marker='o')
plt.plot([p[0] for p in predicted_trajectory], label="Predicted x", linestyle='--', marker='x')
plt.plot([a[1] for a in actual_trajectory], label="Actual z", marker='o')
plt.plot([p[1] for p in predicted_trajectory], label="Predicted z", linestyle='--', marker='x')
plt.xlabel("Time Step")
plt.ylabel("State Value")
plt.title("x and z Evolution Over Time")
plt.legend()
plt.grid(True)
plt.show()

# === Show comparison table ===
df = pd.DataFrame(comparison_table, columns=["Step", "x", "z", "True x'", "True z'", "Pred x'", "Pred z'"])
print(df.head(10).to_string(index=False))  # Show first 10 steps
