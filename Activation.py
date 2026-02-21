
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # Sigmoid function
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

# # Input data
# x = np.linspace(-10, 10, 500)
# target = np.sin(x) * 0.5 + 0.5 * np.cos(2 * x) + 0.2  # target curve

# # Number of neurons (sigmoids) - Increased for better fitting
# n_neurons = 20 # Increased from 6

# # Initialize parameters randomly
# np.random.seed(0)
# slopes = np.random.uniform(-2, 2, n_neurons)
# shifts = np.random.uniform(-5, 5, n_neurons)
# amps = np.random.uniform(-1, 1, n_neurons)

# # Learning rate
# lr = 0.05

# # Prepare plot
# fig, ax = plt.subplots(figsize=(10, 6)) # Slightly larger figure for clarity
# lines_sigmoids = [ax.plot([], [], lw=1, alpha=0.4, color='blue')[0] for _ in range(n_neurons)] # Reduced alpha for many lines
# line_combined, = ax.plot([], [], lw=2, color='red', label='Combined Output')
# line_target, = ax.plot(x, target, 'k--', label='Target Data', alpha=0.8)
# ax.set_xlim(-10, 10)
# ax.set_ylim(-1.5, 1.5)
# ax.set_title("Fitting Complex Shape Using Multiple Sigmoids (Improved Fit)")
# ax.set_xlabel("Input")
# ax.set_ylabel("Output")
# ax.legend()
# ax.grid(True)

# # Update function for animation
# def update(frame):
#     global slopes, shifts, amps

#     # Forward pass
#     outputs = [amps[i] * sigmoid(slopes[i] * (x - shifts[i])) for i in range(n_neurons)]
#     combined = np.sum(outputs, axis=0)

#     # Compute error and gradient descent updates
#     error = combined - target
#     for i in range(n_neurons):
#         s = slopes[i]
#         sh = shifts[i]
#         a = amps[i]
#         y = sigmoid(s * (x - sh)) # Output of the current sigmoid
        
#         # Derivatives with respect to parameters for this sigmoid
#         dy_ds = a * y * (1 - y) * (x - sh)
#         dy_dsh = -a * y * (1 - y) * s
#         dy_da = y

#         # Update parameters using gradient descent
#         slopes[i] -= lr * np.sum(error * dy_ds) / len(x)
#         shifts[i] -= lr * np.sum(error * dy_dsh) / len(x)
#         amps[i]   -= lr * np.sum(error * dy_da) / len(x)

#         # Update individual sigmoid lines
#         lines_sigmoids[i].set_data(x, outputs[i])

#     # Update combined output line
#     line_combined.set_data(x, combined)
#     return lines_sigmoids + [line_combined]

# # Create animation with more frames for proper fit
# ani = animation.FuncAnimation(fig, update, frames=5000, interval=20, blit=True, repeat=False) # Increased frames to 5000

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Sigmoid function
def sigmoid(z):
    # Clip z to prevent overflow/underflow issues with np.exp for very large/small numbers
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Input data
x = np.linspace(-10, 10, 500)
target = np.sin(x) * 0.5 + 0.5 * np.cos(2 * x) + 0.2  # target curve

# Number of neurons (sigmoids) - Significantly Increased for better fitting
n_neurons = 150 # Increased from 50 to 150 for much more capacity

# Initialize parameters randomly
np.random.seed(0)
# Wider range for slopes to allow for very steep or very flat sigmoids
slopes = np.random.uniform(-4, 4, n_neurons) 
# Shifts should cover the entire x-range effectively
shifts = np.random.uniform(-10, 10, n_neurons)
# Wider range for amplitudes to allow for larger or smaller contributions
amps = np.random.uniform(-2, 2, n_neurons)

# Learning rate
lr = 0.02 # Keeping a slightly reduced learning rate

# Prepare plot
fig, ax = plt.subplots(figsize=(14, 8)) # Larger figure for better visualization
# Even lower alpha and thinner lines for many sigmoids to prevent clutter
lines_sigmoids = [ax.plot([], [], lw=0.7, alpha=0.15, color='blue')[0] for _ in range(n_neurons)] 
line_combined, = ax.plot([], [], lw=3, color='red', label='Combined Output') # Thicker combined line
line_target, = ax.plot(x, target, 'k--', label='Target Data', alpha=0.9)
ax.set_xlim(-10, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Fitting Complex Shape Using Multiple Sigmoids (Extensive Fit Attempt)")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.legend()
ax.grid(True)

# Update function for animation
def update(frame):
    global slopes, shifts, amps

    # Forward pass
    outputs = [amps[i] * sigmoid(slopes[i] * (x - shifts[i])) for i in range(n_neurons)]
    combined = np.sum(outputs, axis=0)

    # Compute error and gradient descent updates
    error = combined - target
    for i in range(n_neurons):
        s = slopes[i]
        sh = shifts[i]
        a = amps[i]
        
        # Calculate the output of the current sigmoid
        # Clip the argument to sigmoid to avoid extremely small or large exp values
        z = np.clip(s * (x - sh), -500, 500) 
        y = sigmoid(z)
        
        # Derivatives with respect to parameters for this sigmoid
        # Ensure derivatives are calculated carefully, especially for very flat parts of sigmoid
        dy_dz = y * (1 - y)
        dy_ds = a * dy_dz * (x - sh)
        dy_dsh = -a * dy_dz * s
        dy_da = y

        # Update parameters using gradient descent
        # Use mean of gradients over x to stabilize updates
        # Add a small epsilon to denominator to avoid division by zero if len(x) somehow becomes 0 (though not an issue here)
        slopes[i] -= lr * np.mean(error * dy_ds) 
        shifts[i] -= lr * np.mean(error * dy_dsh)
        amps[i]   -= lr * np.mean(error * dy_da)

        # Update individual sigmoid lines
        lines_sigmoids[i].set_data(x, outputs[i])

    # Update combined output line
    line_combined.set_data(x, combined)
    
    # Update title to show current frame, indicating progress
    ax.set_title(f"Fitting Complex Shape Using Multiple Sigmoids (Frame: {frame}/{20000}) - Error: {np.mean(error**2):.4f}")

    return lines_sigmoids + [line_combined]

# Create animation with significantly more frames for proper fit
ani = animation.FuncAnimation(fig, update, frames=20000, interval=20, blit=True, repeat=False) # Increased frames to 20000

plt.show()

