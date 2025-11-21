"""
Cochlea module: Simulates cochlear filtering and basilar membrane motion.
"""

from .filterbank import apply_filterbank, design_gammatone_filterbank
from .envelope_extract import extract_envelope, extract_instantaneous_phase

__all__ = [
    'apply_filterbank',
    'design_gammatone_filterbank',
    'extract_envelope',
    'extract_instantaneous_phase'
]
