"""Allow ``python -m video_suitability`` as a CLI alias."""

from .infer import main


raise SystemExit(main())
