# Minimal shim for legacy `manufacture.Lasercut`
class _Noop:
    def __init__(self,*a,**k): pass
    def __call__(self,*a,**k): return self
    def __getattr__(self,_):    return self
    def __repr__(self):         return "<Noop>"

class Lasercut:
    def __init__(self, thickness=3.0, kerf=0.1, material="acrylic", **kwargs):
        self.thickness = thickness
        self.kerf = kerf
        self.material = material
        self.parts = []
    # common patterns that old code might call:
    def add(self, *a, **k): self.parts.append((a,k)); return self
    def cut(self, *a, **k): return _Noop()
    def plan(self, *a, **k): return self
    def export(self, *a, **k): return None
    def __repr__(self): return f"<Lasercut t={self.thickness} kerf={self.kerf} material={self.material}>"

class Printable:
    def __init__(self,*a,**k): self.params=dict(k); self.parts=[]
    def add(self,*a,**k): self.parts.append((a,k)); return self
    def plan(self,*a,**k): return self
    def export(self,*a,**k): return None
    def __repr__(self): return f"<Printable {self.params}>"
