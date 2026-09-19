"""torchrun environment adapter for the two-GPU dynamic smoke test."""

import os
import sys

from smoke_helix_fp16_resize_2gpu import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--local-rank", os.environ["LOCAL_RANK"]]
    main()
