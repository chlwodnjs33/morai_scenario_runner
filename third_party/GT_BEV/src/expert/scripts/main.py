#!/usr/bin/env python3
import sys
import os

_pkg_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _pkg_dir)

from src.main import main

if __name__ == '__main__':
    main()
