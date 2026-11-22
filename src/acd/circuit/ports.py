"""Names of the circuit's port nets.

Ports are the nets the test circuit connects to. They can never be deleted or merged with each
other. GND is exported as SPICE node ``0``.
"""

from typing import Final

GND: Final = "GND"
VDD: Final = "VDD"
VIN: Final = "VIN"
VOUT: Final = "VOUT"

DEFAULT_PORTS: Final[tuple[str, ...]] = (VIN, VOUT, VDD, GND)
