"""SPICE device model files shipped with the package."""

from importlib.resources import files


def generic_models() -> str:
    """Generic models used during search: NPN_GEN, PNP_GEN, NMOS_GEN, PMOS_GEN."""
    return (files("acd.models") / "generic.lib").read_text()
