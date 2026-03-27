"""
Surrogate Gradient Demonstration
Shows how surrogate gradients enable SNN training
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================================
# PART 1: Visualize True vs Surrogate Gradient
# ============================================================================

def spike_function(u, V_th=1.0):
    """True spike function (non-differentiable step function)"""
    return (u > V_th).astype(float)

def true_gradient(u, V_th=1.0, epsilon=0.01):
    """
    True derivative of spike function
    (0 everywhere except undefined at threshold)
    """
    grad = np.zeros_like(u)
    grad[(u >= V_th - epsilon) & (u <= V_th + epsilon)] = np.nan  # undefined at threshold
    return grad

def surrogate_gradient(u, V_th=1.0):
    """Surrogate gradient: smooth approximation"""
    return 1.0 / (1.0 + np.abs(u - V_th)**2)**2

# Create visualization
u = np.linspace(-2, 3, 1000)
V_th = 1.0

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Spike Function
ax = axes[0]
spikes = spike_function(u, V_th)
ax.plot(u, spikes, 'b-', linewidth=3, label='Spike function')
ax.axvline(V_th, color='r', linestyle='--', linewidth=2, label=f'Threshold (V_th={V_th})')
ax.fill_between(u, 0, spikes, alpha=0.3)
ax.set_xlabel('Membrane potential (u)', fontsize=12)
ax.set_ylabel('Output spike', fontsize=12)
ax.set_title('FORWARD PASS\nSpike Function (Binary)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.2)

# Plot 2: True Gradient (Useless)
ax = axes[1]
ax.axhline(0, color='r', linewidth=3, label='True gradient (always 0)')
ax.axvline(V_th, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax.scatter([V_th], [np.nan], color='red', s=200, marker='x', linewidth=3, label='Undefined at V_th')
ax.set_xlabel('Membrane potential (u)', fontsize=12)
ax.set_ylabel('Gradient', fontsize=12)
ax.set_title('BACKWARD PASS (❌ PROBLEM)\nTrue Gradient = 0 (No Learning!)',
             fontsize=13, fontweight='bold', color='red')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.5)
ax.text(0.5, 1.2, '❌ Backprop impossible\n❌ Weights cannot update',
        transform=ax.transAxes, fontsize=11, color='red',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Plot 3: Surrogate Gradient (Solution)
ax = axes[2]
surr_grad = surrogate_gradient(u, V_th)
ax.plot(u, surr_grad, 'g-', linewidth=3, label='Surrogate gradient')
ax.axvline(V_th, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax.fill_between(u, 0, surr_grad, alpha=0.3, color='green')
ax.set_xlabel('Membrane potential (u)', fontsize=12)
ax.set_ylabel('Gradient', fontsize=12)
ax.set_title('BACKWARD PASS (✅ SOLUTION)\nSurrogate Gradient (Smooth)',
             fontsize=13, fontweight='bold', color='green')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.85, '✅ Smooth gradient\n✅ Backprop works!',
        transform=ax.transAxes, fontsize=11, color='green',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('surrogate_gradient_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: surrogate_gradient_comparison.png")
plt.show()

# ============================================================================
# PART 2: Train a Simple SNN with Surrogate Gradient
# ============================================================================

class SimpleSNNNeuron:
    """Single SNN neuron with surrogate gradient"""

    def __init__(self, input_size=2, learning_rate=0.01, V_th=1.0):
        self.W = np.random.randn(input_size) * 0.1  # Weights
        self.b = np.random.randn() * 0.1              # Bias
        self.lr = learning_rate
        self.V_th = V_th
        self.u = 0  # Membrane potential

    def forward(self, x):
        """Forward pass: compute spike"""
        self.u = np.dot(x, self.W) + self.b
        spike = float(self.u > self.V_th)  # Binary output
        return spike, self.u

    def backward_with_surrogate(self, x, target, spike):
        """Backward pass: use surrogate gradient"""
        # Error
        error = spike - target

        # Surrogate gradient at current u
        grad_surrogate = surrogate_gradient(np.array([self.u]), self.V_th)[0]

        # Gradient w.r.t weights
        dW = error * grad_surrogate * x
        db = error * grad_surrogate

        # Update weights
        self.W -= self.lr * dW
        self.b -= self.lr * db

        return error

# Train on simple task: "output 1 if sum(input) > 1.5, else 0"
print("\n" + "="*70)
print("TRAINING SIMPLE SNN WITH SURROGATE GRADIENT")
print("="*70)

neuron = SimpleSNNNeuron(input_size=2, learning_rate=0.1)

# Training data
X_train = np.array([
    [0.5, 0.5],
    [0.2, 0.2],
    [2.0, 1.0],
    [1.5, 1.0],
    [-0.5, 0.5],
    [2.5, 2.0],
])

y_train = np.array([
    0,  # sum=1.0 < 1.5
    0,  # sum=0.4 < 1.5
    1,  # sum=3.0 > 1.5
    1,  # sum=2.5 > 1.5
    0,  # sum=0.0 < 1.5
    1,  # sum=4.5 > 1.5
])

losses = []
accuracies = []

for epoch in range(100):
    total_loss = 0
    correct = 0

    for x, target in zip(X_train, y_train):
        spike, u = neuron.forward(x)
        error = neuron.backward_with_surrogate(x, target, spike)
        total_loss += error**2
        correct += (spike == target)

    avg_loss = total_loss / len(X_train)
    accuracy = correct / len(X_train)
    losses.append(avg_loss)
    accuracies.append(accuracy)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1%}")

print(f"Final:   Loss={losses[-1]:.4f}, Accuracy={accuracies[-1]:.1%}")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(losses, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('MSE Loss', fontsize=12)
axes[0].set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(accuracies, 'g-', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Classification Accuracy Over Time', fontsize=13, fontweight='bold')
axes[1].set_ylim([0, 1.05])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('snn_training_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: snn_training_curves.png")
plt.show()

# ============================================================================
# PART 3: Visualize Decision Boundary
# ============================================================================

print("\n" + "="*70)
print("DECISION BOUNDARY AFTER TRAINING")
print("="*70)

# Create grid for visualization
x_range = np.linspace(-1, 3, 100)
y_range = np.linspace(-1, 3, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

Z = np.zeros_like(X_grid)
for i in range(X_grid.shape[0]):
    for j in range(X_grid.shape[1]):
        x = np.array([X_grid[i, j], Y_grid[i, j]])
        spike, _ = neuron.forward(x)
        Z[i, j] = spike

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Decision boundary
contour = ax.contourf(X_grid, Y_grid, Z, levels=[0, 0.5, 1], colors=['lightblue', 'lightgreen'], alpha=0.6)
ax.contour(X_grid, Y_grid, Z, levels=[0.5], colors=['red'], linewidths=2)

# Training data points
colors = ['red' if y == 0 else 'green' for y in y_train]
ax.scatter(X_train[:, 0], X_train[:, 1], c=colors, s=200, edgecolors='black', linewidth=2, zorder=5)

# Add labels
for i, (x, y) in enumerate(X_train):
    ax.annotate(f'{y_train[i]}', xy=(x, y), xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

# Ground truth boundary (sum = 1.5)
x_line = np.linspace(-1, 3, 100)
y_line = 1.5 - x_line
ax.plot(x_line, y_line, 'k--', linewidth=2, label='Ideal boundary (sum=1.5)')

ax.set_xlabel('Input 1', fontsize=12)
ax.set_ylabel('Input 2', fontsize=12)
ax.set_title('Learned Decision Boundary\n(Red=Class 0, Green=Class 1)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)

plt.tight_layout()
plt.savefig('snn_decision_boundary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: snn_decision_boundary.png")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY: WHY SURROGATE GRADIENTS WORK")
print("="*70)
print("""
✅ FORWARD PASS:  Use true spike function (binary: 0 or 1)
   → Energy efficient, sparse activations (like biological neurons)

❌ PROBLEM:        True gradient = 0 everywhere (no learning!)

✅ BACKWARD PASS:  Use smooth surrogate gradient for weight updates
   → Enables backpropagation despite binary spikes

🎯 RESULT:         SNNs can be trained end-to-end with backprop
   → Maintain spike sparsity AND learn effectively
   → Best of both worlds!

📊 This demo showed:
   1. Visual comparison: True gradient (useless) vs Surrogate (smooth)
   2. Training curve: SNN learns using surrogate gradient
   3. Decision boundary: Neuron learns meaningful classification
""")
print("="*70)
