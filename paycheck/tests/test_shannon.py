from biom import Table
from numpy import array, log
from numpy.testing import assert_almost_equal

from paycheck.shannon import get_stats


def test_get_stats():
    def entropy(p):
        p = p[p != 0]
        return -(p*log(p)).sum()

    data = array([[0, 0, 1], [1, 3, 42]], dtype=float)
    table = Table(data, ['O1', 'O2'], ['S1', 'S2', 'S3'])
    h, jsd = get_stats(table)

    p = data.sum(axis=1)
    p /= p.sum()
    assert_almost_equal(h, entropy(p))
    avg_h = 0
    weights = [p.sum()/data.sum() for p in data.T]
    for w, p in zip(weights, data.T):
        p /= p.sum()
        avg_h += w*entropy(p)
    assert_almost_equal(jsd, h - avg_h)

    table.norm()
    h, jsd = get_stats(table)

    data /= data.sum(axis=0)
    p = data.sum(axis=1) / 3.
    assert_almost_equal(h, entropy(p))
    avg_h = sum(entropy(p)/3. for p in data.T)
    assert_almost_equal(jsd, h - avg_h)
