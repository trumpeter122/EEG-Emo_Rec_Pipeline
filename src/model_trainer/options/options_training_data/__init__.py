"""
Training-data option registry placeholder.

``TrainingDataOption`` eagerly loads extracted feature files.  To avoid large
imports during CLI startup, instantiate those options dynamically within
execution scripts (e.g., see ``main.py``) rather than pre-registering them
here.  This module remains as a documentation anchor for contributors.
"""

__all__: list[str] = []
