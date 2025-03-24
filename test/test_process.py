import numpy as np
import dnasufo


def test_process():
    """Simple smoke test"""
    data = dnasufo.example([50, 60, 60])
    pimg, cell_lbl, cell_trj, cell_flow, dna_lbl, dna_trj, dna_flow = dnasufo.process(
        data
    )
    assert np.array_equal(data.shape, pimg.shape)
