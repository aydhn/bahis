import re

file_path = "src/extensions/regime_hmm.py"
with open(file_path, "r") as f:
    content = f.read()

new_train_method = """    def train(self, observations: np.ndarray, alpha: float = 0.1):
        \"\"\"
        Online EMA training.
        Dynamically updates the means and vars based on observations without full EM overhead.
        \"\"\"
        if len(observations) == 0:
            return

        # Get current state prediction
        prediction = self.predict(observations)
        curr_state = prediction.current_state

        for obs in observations:
            diff = obs - self.means[curr_state]
            self.means[curr_state] += alpha * diff
            self.vars[curr_state] = (1 - alpha) * self.vars[curr_state] + alpha * (diff ** 2)
            self.vars[curr_state] = max(self.vars[curr_state], 1e-6)

        logger.debug(f"HMM parameters updated. New Means: {self.means}")
"""

# Let's use a simpler replacement to be safe.
# Find the exact string
old_string = """    def train(self, observations: np.ndarray):
        \"\"\"
        Stub for Baum-Welch training.
        In production, this would update A, means, and vars based on history.
        \"\"\"
        pass"""

if old_string in content:
    content = content.replace(old_string, new_train_method)
    with open(file_path, "w") as f:
        f.write(content)
    print("regime_hmm.py patched successfully.")
else:
    print("Old string not found. Maybe already patched?")
