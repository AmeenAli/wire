"""WIRE: Weight Space Implicit Representation Encoding of Frozen Video LLMs."""

from .config import WireConfig
from .loading import load_wire, load_wire_weights, read_wire_state_dict
from .modeling import (attach_wire, controls, merge_updates, set_prompt_end,
                       unmerge_updates)

__all__ = [
    "WireConfig", "attach_wire", "controls", "load_wire", "load_wire_weights",
    "merge_updates", "read_wire_state_dict", "set_prompt_end",
    "unmerge_updates",
]
