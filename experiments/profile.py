import contextlib
from time import perf_counter as now


class Profile(contextlib.ContextDecorator):
    def __init__(self, t=0.0, pmean=0.5, pvar=0.25, pfnum=1, item=None):
        self.dt = t
        self.obj = item
        self.pmean= pmean
        self.pvar = pvar
        self.pfnum = pfnum

    def __enter__(self):
        self.start = now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dt = now() - self.start
        if self.obj is not None:
            self.obj.reset(self.pmean, self.pvar, self.pfnum)
