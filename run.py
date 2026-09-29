"""실행: python run.py [--sim | --hw] [--config config.yaml]"""
import sys

from paxtest.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
