"""Domain loaders, one module per data source family.

Each module is the block that used to be a section of services/data_loader.py, moved without
editing. The facade in that module re-exports every public and private name, so existing imports
(`services.data_loader._clean_numeric`, `data_loader.load_crime`, ...) are unchanged."""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from services.loaders import redfin  # noqa: F401
from services.loaders import nri  # noqa: F401
from services.loaders import crime  # noqa: F401
from services.loaders import census  # noqa: F401
from services.loaders import cbsa  # noqa: F401
from services.loaders import sold  # noqa: F401
from services.loaders import bike  # noqa: F401
from services.loaders import zillow  # noqa: F401
from services.loaders import transit  # noqa: F401

from services.loaders.redfin import *  # noqa: F401,F403
from services.loaders.nri import *  # noqa: F401,F403
from services.loaders.crime import *  # noqa: F401,F403
from services.loaders.census import *  # noqa: F401,F403
from services.loaders.cbsa import *  # noqa: F401,F403
from services.loaders.sold import *  # noqa: F401,F403
from services.loaders.bike import *  # noqa: F401,F403
from services.loaders.zillow import *  # noqa: F401,F403
from services.loaders.transit import *  # noqa: F401,F403

__all__ = sorted(
    {
        *redfin.__all__,
        *nri.__all__,
        *crime.__all__,
        *census.__all__,
        *cbsa.__all__,
        *sold.__all__,
        *bike.__all__,
        *zillow.__all__,
        *transit.__all__,
    }
)
