from src.quant.physics.entropy_meter import shannon_entropy
import numpy as np

probs = np.array([0.5, 0.5])
print(shannon_entropy(probs))
