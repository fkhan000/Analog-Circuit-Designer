"""Graph-based crossover: replace a connected subgraph of one parent with one from the other.

1. Grow a connected region of 1..``max_region_size`` components in parent A. Growth follows
   internal nets and the signal ports (VIN, VOUT) but not GND/VDD: the rails touch almost
   everything, so growing through them would make regions arbitrary instead of local.
2. The region's *boundary nets* are internal nets it shares with the rest of A. Port nets it touches
   keep their identity (VDD stays VDD), so they need no mapping.
3. Grow a region in parent B, preferring one with the same number of boundary nets.
4. Remove A's region, add B's region with its boundary nets mapped onto A's boundary nets, and keep
   the first mapping that passes the graph checks. B's interior nets become fresh nets.
"""

from __future__ import annotations

import itertools
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from acd.circuit.graph import Circuit
from acd.circuit.ports import GND, VDD
from acd.circuit.validity import PortContext, check

# Nets that regions don't grow through.
_RAILS = frozenset({GND, VDD})


@dataclass(frozen=True)
class CrossoverConfig:
    max_region_size: int = 4  # components in a swapped region
    max_attempts: int = 10  # region pairs tried per crossover
    region_tries: int = 8  # regions grown in parent B looking for a matching boundary count
    max_mappings: int = 24  # boundary mappings tried per region pair


@dataclass(frozen=True)
class Region:
    components: tuple[str, ...]  # component ids, in circuit order
    boundary: tuple[str, ...]  # internal nets shared with components outside the region
    interior: tuple[str, ...]  # internal nets used only inside the region


@dataclass
class CrossoverResult:
    circuit: Circuit | None
    attempts: int = 0
    rejections: Counter[str] = field(default_factory=Counter)


def region_of(circuit: Circuit, ids: set[str]) -> Region:
    nets = {n for c in circuit.components if c.id in ids for n in c.nets}
    boundary, interior = [], []
    for n in sorted(nets - set(circuit.ports)):
        if all(cid in ids for cid, _ in circuit.connections(n)):
            interior.append(n)
        else:
            boundary.append(n)
    order = tuple(c.id for c in circuit.components if c.id in ids)
    return Region(order, tuple(boundary), tuple(interior))


def grow_region(circuit: Circuit, size: int, rng: np.random.Generator) -> Region:
    """A connected region of up to ``size`` components, grown breadth-first in random order."""
    start = circuit.components[int(rng.integers(len(circuit)))].id
    chosen = {start}
    queue = deque([start])
    while queue and len(chosen) < size:
        comp = circuit.component(queue.popleft())
        neighbors = sorted(
            {
                cid
                for n in comp.nets
                if n not in _RAILS
                for cid, _ in circuit.connections(n)
                if cid not in chosen
            }
        )
        for i in rng.permutation(len(neighbors)):
            if len(chosen) >= size:
                break
            chosen.add(neighbors[int(i)])
            queue.append(neighbors[int(i)])
    return region_of(circuit, chosen)


def _mappings(
    src: tuple[str, ...],
    dst: tuple[str, ...],
    fallback: tuple[str, ...],
    rng: np.random.Generator,
    limit: int,
) -> Iterator[dict[str, str]]:
    """Ways to map B's boundary nets onto A's.

    Equal counts: permutations (in random order). Otherwise each B net picks a random A boundary
    net, or any net of the child when A's region had no boundary nets.
    """
    if not src:
        yield {}
        return
    if len(src) == len(dst):
        perms = list(itertools.permutations(dst))
        for i in rng.permutation(len(perms))[:limit]:
            yield dict(zip(src, perms[int(i)], strict=True))
        return
    pool = dst or fallback
    for _ in range(min(limit, len(pool) ** len(src))):
        yield {s: pool[int(rng.integers(len(pool)))] for s in src}


def _region_size(circuit: Circuit, config: CrossoverConfig, rng: np.random.Generator) -> int:
    return int(rng.integers(1, min(config.max_region_size, len(circuit)) + 1))


def subgraph_swap(
    a: Circuit,
    b: Circuit,
    rng: np.random.Generator,
    config: CrossoverConfig | None = None,
    max_components: int = 10,
    ctx: PortContext | None = None,
) -> CrossoverResult:
    """A child of ``a`` with one of its regions replaced by a region of ``b``."""
    config = config or CrossoverConfig()
    rejections: Counter[str] = Counter()
    if len(a) == 0 or len(b) == 0:
        rejections["empty_parent"] += 1
        return CrossoverResult(None, 0, rejections)
    if set(b.ports) - set(a.ports):
        raise ValueError("parents must have the same ports")

    for attempt in range(1, config.max_attempts + 1):
        ra = grow_region(a, _region_size(a, config, rng), rng)
        rb = grow_region(b, _region_size(b, config, rng), rng)
        for _ in range(config.region_tries - 1):
            if len(rb.boundary) == len(ra.boundary):
                break
            rb = grow_region(b, _region_size(b, config, rng), rng)
        if len(a) - len(ra.components) + len(rb.components) > max_components:
            rejections["too_many_components"] += 1
            continue

        base = a
        for cid in ra.components:
            base = base.remove(cid)
        fresh: dict[str, str] = {}
        for n in rb.interior:
            base, fresh[n] = base.new_net()
        donors = [b.component(cid) for cid in rb.components]

        for mapping in _mappings(rb.boundary, ra.boundary, base.nets, rng, config.max_mappings):
            child = base
            for comp in donors:
                nets = tuple(mapping.get(n) or fresh.get(n) or n for n in comp.nets)
                child, _ = child.add(comp.kind, nets, comp.params)
            violations = check(child, ctx)
            if not violations:
                return CrossoverResult(child, attempt, rejections)
            rejections.update(str(v.kind) for v in violations)
    return CrossoverResult(None, config.max_attempts, rejections)
