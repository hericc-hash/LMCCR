"""Final frozen R3.2-S4-v2.2 selector."""
from .firewall import inference_features, require_final_selector
from .selector import FrozenSelector

__all__ = ['FrozenSelector', 'inference_features', 'require_final_selector']
