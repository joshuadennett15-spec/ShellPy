"""
Europa2D — axisymmetric latitude-column extension of the 1D thermal-shell
model. The 1D modules on which this package depends live in the sibling
``Europa1D/src`` directory; the path setup below makes them importable under
their original names.
"""
import os
import sys

_EUROPA1D_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'Europa1D', 'src')
)
if os.path.isdir(_EUROPA1D_SRC) and _EUROPA1D_SRC not in sys.path:
    sys.path.insert(0, _EUROPA1D_SRC)
