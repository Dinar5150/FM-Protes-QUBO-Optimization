from __future__ import annotations

import time
from typing import Any

from dwave.samplers import SimulatedAnnealingSampler


def DWaveQuboSolver(Q: Any, num_reads: int = 1000):
    """Solve a QUBO with D-Wave's simulated annealing sampler (dwave-samplers).

    Returns a dimod SampleSet.
    """
    t0 = time.time()
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=int(num_reads))
    print("total time taken:", time.time() - t0)
    print(sampleset.info)
    print("num_reads executed:", sampleset.record.num_occurrences.sum())
    return sampleset
