"""Legacy 1.18c planning and P-only realization for auxiliary compatibility.

The final factual Planner is lumbar_cf_report.planner.Stage23UnifiedClinicalPlanner.
"""

from .planner import ExplicitClinicalPlanner
from .realizer import PlanOnlyReportRealizer

__all__ = ["ExplicitClinicalPlanner", "PlanOnlyReportRealizer"]

