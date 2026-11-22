"""Structural hashes for caching and duplicate detection.

Hashes ignore component ids, internal net names and the pin order of R, C and L. They reflect
component types, connections, which port is which and (for ``full_hash``) component values.

The Weisfeiler-Lehman hash can, rarely, give two different circuits the same hash. Anything that
must be exact (such as the evaluation cache) should confirm a hash match with ``isomorphic``.
"""

from __future__ import annotations

import networkx as nx

from acd.circuit.devices import spec
from acd.circuit.graph import Circuit

_WL_ITERATIONS = 5

Node = tuple[str, str]  # ("c", component id) or ("n", net name)


def to_graph(circuit: Circuit, *, with_values: bool) -> nx.Graph[Node]:
    """The bipartite graph: component nodes, net nodes, edges labeled with pin names."""
    g: nx.Graph[Node] = nx.Graph()
    for c in circuit.components:
        label = str(c.kind)
        if with_values:
            label += "|" + ",".join(f"{v:.6g}" for v in c.params)
        g.add_node(("c", c.id), label=label)
    for net in circuit.nets:
        g.add_node(("n", net), label=f"port:{net}" if circuit.is_port(net) else "net")
    for c in circuit.components:
        s = spec(c.kind)
        # Pins of one component on the same net become a single edge with a combined label
        # (e.g. a diode-connected BJT gets one "b+c" edge), so a simple graph is enough.
        labels: dict[str, list[str]] = {}
        for pin, net in zip(s.pin_names, c.nets, strict=True):
            labels.setdefault(net, []).append("x" if s.symmetric else pin)
        for net, pin_labels in labels.items():
            g.add_edge(("c", c.id), ("n", net), label="+".join(sorted(pin_labels)))
    return g


def _wl(g: nx.Graph[Node]) -> str:
    return str(
        nx.weisfeiler_lehman_graph_hash(
            g, node_attr="label", edge_attr="label", iterations=_WL_ITERATIONS
        )
    )


def topology_hash(circuit: Circuit) -> str:
    """Hash of the structure only (component values ignored)."""
    return _wl(to_graph(circuit, with_values=False))


def full_hash(circuit: Circuit) -> str:
    """Hash of the structure and the component values."""
    return _wl(to_graph(circuit, with_values=True))


def isomorphic(a: Circuit, b: Circuit, *, with_values: bool = True) -> bool:
    """Exact check that two circuits are the same up to renaming."""
    if len(a) != len(b):
        return False
    ga, gb = to_graph(a, with_values=with_values), to_graph(b, with_values=with_values)
    return bool(
        nx.is_isomorphic(
            ga,
            gb,
            node_match=lambda x, y: x["label"] == y["label"],
            edge_match=lambda x, y: x["label"] == y["label"],
        )
    )
