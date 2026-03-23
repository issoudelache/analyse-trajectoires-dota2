"""Pages de l'interface graphique."""

from mvc.views.pages.base_page import BasePage
from mvc.views.pages.menu_page import MenuPage
from mvc.views.pages.overlay_page import OverlayPage
from mvc.views.pages.compress_page import CompressPage
from mvc.views.pages.cluster_page import ClusterPage
from mvc.views.pages.comparison_page import ComparisonPage
from mvc.views.pages.mining_page import MiningPage

__all__ = [
    "BasePage",
    "MenuPage",
    "OverlayPage",
    "CompressPage",
    "ClusterPage",
    "ComparisonPage",
    "MiningPage",
]
