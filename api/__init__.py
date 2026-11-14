from api.houses import router as houses_router
from api.chat import router as chat_router
from api.bike import router as bike_router
from api.system import router as system_router
from api.commute import router as commute_router
from api.data_sources import router as data_sources_router
from api.map_layers import router as map_layers_router
from api.onboarding import router as onboarding_router

__all__ = [
    "houses_router",
    "chat_router",
    "bike_router",
    "system_router",
    "commute_router",
    "data_sources_router",
    "map_layers_router",
    "onboarding_router",
]

