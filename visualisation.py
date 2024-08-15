import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# This function assigns a unique color based on position
def scatter_colorizer(x, y):
    """
    Map x-y coordinates to a rgb color
    """
    r = min(1, 1-y/3)
    g = min(1, 1+y/3)
    b = 1/4 + x/16
    return (r, g, b)





# Define a square with 4 points (clockwise)
square = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0]  # Repeat the first point to close the square
])

def apply_transformation(points, matrix):
    """Apply an affine transformation to a set of points."""
    transformed_points = np.dot(points, matrix.T)
    return transformed_points

def plot_square(ax, points, title, color='blue'):
    """Plot a square given its points."""
    ax.plot(points[:, 0], points[:, 1], color=color, lw=2)
    ax.fill(points[:, 0], points[:, 1], color=color, alpha=0.3)
    ax.set_title(title)
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.grid(True)
    ax.set_aspect('equal')

# Define a set of 2D affine transformation matrices
transformations = {
    "Original": np.array([[1, 0], [0, 1]]),
    "Translation (1, 1)": np.array([[1, 0], [0, 1]]),  # Translation will be handled separately
    "Scaling (2x, 0.5x)": np.array([[2, 0], [0, 0.5]]),
    "Rotation (45°)": np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], 
                                [np.sin(np.pi/4), np.cos(np.pi/4)]]),
    "Shearing (x)": np.array([[1, 0.5], [0, 1]])
}

# Plot each transformation
fig, axs = plt.subplots(1, len(transformations), figsize=(15, 5))

for i, (name, matrix) in enumerate(transformations.items()):
    if name == "Translation (1, 1)":
        transformed_square = square + np.array([1, 1])  # Translate manually
    else:
        transformed_square = apply_transformation(square, matrix)
    
    plot_square(axs[i], transformed_square, name)

plt.tight_layout()
plt.show()