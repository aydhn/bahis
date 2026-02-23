import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.cli.commands import app

def main():
    app()

if __name__ == "__main__":
    main()
