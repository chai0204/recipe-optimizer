"""Output formatters: produce the three PoC views over a scheduled DAG.

- ``numbered_list``: chronological numbered steps (plain text)
- ``dag_viz``: structured DAG visualization (Mermaid syntax)
- ``shopping_list``: consolidated ingredients + tools list
"""

from .dag_viz import to_mermaid
from .numbered_list import to_numbered_steps
from .shopping_list import aggregate_tools, build_shopping_list

__all__ = [
    "to_numbered_steps",
    "to_mermaid",
    "aggregate_tools",
    "build_shopping_list",
]
