"""Enable ``python -m bsf`` / ``torchrun ... -m bsf.cli``."""
from .cli import main

if __name__ == '__main__':
    raise SystemExit(main())
