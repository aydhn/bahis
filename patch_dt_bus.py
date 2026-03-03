import re

with open("src/system/digital_twin.py", "r") as f:
    content = f.read()

pattern = r'bus = container\.get\("bus"\)'
replaced = re.sub(pattern, 'bus = container.get("bus", default=None)', content)

with open("src/system/digital_twin.py", "w") as f:
    f.write(replaced)
