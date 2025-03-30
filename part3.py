import numpy as np
from sklearn.model_selection import train_test_split
import RegressionTree as RT
import matplotlib.pyplot as plt

# Generate dataset for the dynamical system
def generate_dynamical_data(n_samples=500):
    x1 = np.random.uniform(-5, 5, n_samples)
    x2 = np.random.uniform(-5, 5, n_samples)
    x1_next = 0.9 * x1 - 0.2 * x2
    x2_next = 0.2 * x1 + 0.9 * x2
    X = np.column_stack((x1, x2))
    y = np.column_stack((x1_next, x2_next))
    return X, y

# Generate data
X, y = generate_dynamical_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RT.RegressionTree(X_train, limit_type=None)

predictions = model.predict(model, X_test)

initial_state = np.array([0.5, 1.5]).reshape(1, -1)
predicted_trajectory = [initial_state.flatten()]
actual_trajectory = [initial_state.flatten()]

for _ in range(20):
    next_pred = model.predict(initial_state).flatten()
    next_actual = np.array([0.9 * initial_state[0, 0] - 0.2 * initial_state[0, 1],
                             0.2 * initial_state[0, 0] + 0.9 * initial_state[0, 1]])
    predicted_trajectory.append(next_pred)
    actual_trajectory.append(next_actual)
    initial_state = next_actual.reshape(1, -1)

predicted_trajectory = np.array(predicted_trajectory)
actual_trajectory = np.array(actual_trajectory)

plt.figure(figsize=(8, 6))
plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], 'bo-', label='Actual Trajectory')
plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'ro--', label='Predicted Trajectory')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Dynamical System Approximation with Regression Tree')
plt.legend()
plt.show()