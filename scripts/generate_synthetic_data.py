"""Keep the original helper entrypoint compatible with the packaged generator."""

from src.synthetic_data import (
    build_parser as build_parser,
)
from src.synthetic_data import (
    generate_synthetic_dataset as generate_synthetic_dataset,
)
from src.synthetic_data import (
    main as main,
)
from src.synthetic_data import (
    write_synthetic_dataset as write_synthetic_dataset,
)

if __name__ == "__main__":
    raise SystemExit(main())
