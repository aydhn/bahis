import re

with open("src/reporting/visualizer.py", "r") as f:
    code = f.read()

replacement = """try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALS_ENABLED = True
except ImportError:
    VISUALS_ENABLED = False"""

code = code.replace("""import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns""", replacement)

with open("src/reporting/visualizer.py", "w") as f:
    f.write(code)

print("Patched visualizer.py")
