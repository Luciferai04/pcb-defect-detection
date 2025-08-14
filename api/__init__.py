"""
PCB Defect Detection API Package
===============================

This package provides API interfaces for PCB defect detection, including:
- FastAPI web service (main.py)
- Command-line inference tool (inference.py)
- Batch processing utilities
- Model serving capabilities

Example usage:
    # Web API
    from api import app
    
    # Command-line inference
    from api.inference import PCBDefectInference
    
    inference = PCBDefectInference()
    result = inference.predict_single("image.jpg")
"""

from .inference import PCBDefectInference

__version__ = "1.0.0"
__all__ = ["PCBDefectInference"]
