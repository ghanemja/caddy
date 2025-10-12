from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class PartRef:
    """
    Minimal stand-in for legacy `partref.PartRef`.
    Stores metadata about a purchased/standard part (id, vendor, url, params).
    Extend if your code asks for more.
    """
    name: str = ""
    id: Optional[str] = None
    vendor: Optional[str] = None
    url: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = dict(
            name=self.name,
            id=self.id,
            vendor=self.vendor,
            url=self.url,
        )
        d.update(self.params or {})
        return d

    def get(self, key: str, default: Any = None) -> Any:
        # convenience accessor some code uses
        return (self.params or {}).get(key, default)

    def __repr__(self) -> str:
        base = f"PartRef(name={self.name!r}"
        if self.id: base += f", id={self.id!r}"
        if self.vendor: base += f", vendor={self.vendor!r}"
        if self.url: base += f", url={self.url!r}"
        if self.params: base += f", params={self.params!r}"
        return base + ")"
