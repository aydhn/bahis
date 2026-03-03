import re

with open("src/system/digital_twin.py", "r") as f:
    content = f.read()

pattern = r'bus = container\.get\("bus", default=None\)'
replaced = re.sub(pattern, 'try:\n                bus = container.get("bus")\n            except:\n                bus = None', content)

with open("src/system/digital_twin.py", "w") as f:
    f.write(replaced)
