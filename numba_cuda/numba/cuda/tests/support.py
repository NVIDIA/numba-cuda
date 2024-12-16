from numba.cuda.runtime.nrt import rtsys


class EnableNRTStatsMixin(object):
    """Mixin to enable the NRT statistics counters."""

    def setUp(self):
        rtsys.memsys_enable_stats()

    def tearDown(self):
        rtsys.memsys_disable_stats()
