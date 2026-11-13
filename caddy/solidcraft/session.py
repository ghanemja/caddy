"""A caddy session whose spec is a picture and whose modeller is SolidCraft.

Claude drives SolidCraft's MCP server (sketches, constraints, a parametric feature timeline). After every change
caddy exports the design, rebuilds it as SMLib solids and judges it: the same build → check → repair loop as every
other caddy session, with a `.sc` design SolidCraft can open at the end.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import tempfile
import uuid
from pathlib import Path

from smlibquery import kernel as sq_kernel

from caddy.agent.session import Session
from caddy.kernel import graph as G
from caddy.kernel.family import SizeReport, build_size

from . import judge
from .mcp import Mcp, McpError, find_cli

# SolidCraft's own tools Claude may call, with their real schemas (read from the server, so they follow its version).
# Files stay caddy's business: open / save / export / new_design are not offered.
PASS_THROUGH = ("skill_reference", "inspect_design", "batch", "execute", "measure", "body_topology", "set_parameter",
                "check_parameters", "screenshot", "undo", "redo", "list_commands", "describe_command")
CHANGES_DESIGN = {"batch", "execute", "set_parameter", "undo", "redo"}
MAX_TOOL_TEXT = 24000            # characters of one SolidCraft reply kept for the model

EXT = {"image/png": "png", "image/jpeg": "jpg"}


MAX_IMAGE_BYTES = 5_000_000      # the LLM API's limit for one picture


def sniff(data: bytes) -> str:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("the picture is over 5 MB: use a smaller one")
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    raise ValueError("the picture must be a PNG or a JPEG")


class ImageSession(Session):
    def __init__(self, image: bytes, hint: str = "", workdir: str | Path | None = None, cli: str | None = None,
                 min_overlap: float | None = None, min_parameters: int = 3, title: str = "Image → CAD"):
        self.family = "image"
        self.id = uuid.uuid4().hex[:8]
        self.mime = sniff(image)
        self.image, self.hint = image, hint.strip()
        self.dir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="caddy-image-"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.reference = self.dir / f"reference.{EXT[self.mime]}"
        self.reference.write_bytes(image)
        self.min_overlap = float(os.environ.get("CADDY_IMAGE_MIN_OVERLAP", 0.7)) if min_overlap is None else min_overlap
        self.min_parameters = min_parameters
        self.declared: dict | None = None      # what Claude says the object is, its symmetry plane, the picture's view
        self.revision = None                   # SolidCraft's document revision at the last judgement
        self.sc_parameters: list[dict] = []
        self.saved: dict = {}
        self.facts = judge.FACTS.setdefault(self.id, {})
        self.spec = dict(family="image", title=title, size_name="image", required_aliases=[], checks=[],
                         material={"name": "unspecified", "density_g_cm3": 1.0})
        self.rows = [dict(size="image")]
        self.row_names = set(G.ROW_NAMES)
        self.ev = G.Evaluator()
        self.graph = None
        self.reports = []
        self.last_stab = None
        self.versions = []
        self.locked = set()
        self.script = None
        if self.dir not in sq_kernel.STOCK_PATHS:
            sq_kernel.STOCK_PATHS.append(self.dir)
        cli = cli or find_cli()
        if not cli:
            raise McpError("solidcraft-cli was not found: install SolidCraft (https://solidcraft.io/download) or set "
                           "SOLIDCRAFT_CLI to the path of solidcraft-cli")
        self.mcp = Mcp([cli, "mcp", "--headless", "--compact", "--allow-path", str(self.dir)])
        self._tools = {t["name"]: t for t in self.mcp.tools()}
        self._set_checks()

    def close(self) -> None:
        self.mcp.close()
        judge.FACTS.pop(self.id, None)

    # --- the spec: what "done" is measured against -------------------------------------------------------
    def _set_checks(self, sym_tol: float = 0.05) -> None:
        user = [c for c in self.spec["checks"] if c.get("by") == "you"]
        d = self.declared or {}
        plane = d.get("symmetry_plane")
        checks = [
            dict(id="export_intact", measure="image_volume_gap", image=self.id, op="<=", expect="0.005", tol=0.0,
                 rule="the solids SMLib reads back from the STEP export have the volume SolidCraft reports, within 0.5 %"),
            dict(id="features_healthy", measure="image_fact", fact="feature_errors", image=self.id, op="<=",
                 expect="0", tol=0.0, rule="no feature in SolidCraft's timeline reports an error"),
            dict(id="parametric", measure="image_fact", fact="parameters", image=self.id, op=">=",
                 expect=str(self.min_parameters), tol=0.0,
                 rule=f"the design has at least {self.min_parameters} named parameters"),
            dict(id="parameters_drive", measure="image_fact", fact="dead_parameters", image=self.id, op="<=",
                 expect="0", tol=0.0,
                 rule="every named parameter rebuilds without errors and changes the geometry when it changes"),
            dict(id="matches_picture", measure="image_fact", fact="overlap", image=self.id, op=">=",
                 expect=str(self.min_overlap), tol=0.0,
                 rule=f"the model's outline, seen as the picture was taken, overlaps the object's outline in the "
                      f"picture by at least {self.min_overlap:g}"),
        ]
        if plane in judge.AXIS:
            checks.insert(1, dict(id="symmetric", measure="image_symmetry", plane=plane, image=self.id, op="<=",
                                  expect=str(sym_tol), tol=0.0,
                                  rule=f"mirrored across {plane}, every surface point lands back on the model "
                                       f"within {sym_tol:g} mm"))
        elif plane != "none":
            checks.insert(1, dict(id="symmetric", measure="image_fact", fact="undeclared", image=self.id, op="<=",
                                  expect="0", tol=0.0,
                                  rule="the object's mirror plane is declared (describe_object), or the evidence "
                                       "that it has none"))
        self.spec["checks"] = checks + user

    # --- SolidCraft --------------------------------------------------------------------------------------
    def _sc(self, name: str, args: dict | None = None):
        r = self.mcp.call(name, args or {})
        if r.is_error:
            raise G.GraphError(f"SolidCraft {name}: {r.text[:400]}")
        return r

    def _sc_json(self, name: str, args: dict | None = None) -> dict:
        r = self._sc(name, args)
        out = r.json()
        if not isinstance(out, dict):
            raise G.GraphError(f"SolidCraft {name}: unexpected reply {r.text[:200]!r}")
        return out

    def _revision(self):
        try:
            return self._sc_json("execute", {"command": "document.revision"}).get("revision")
        except (G.GraphError, McpError):
            return None

    def tools(self) -> list[dict]:
        """Anthropic tool definitions: SolidCraft's own tools as the server describes them, plus caddy's three."""
        out = []
        for name in PASS_THROUGH:
            t = self._tools.get(name)
            if t:
                out.append(dict(name=name, description=t.get("description", "")[:6000],
                                input_schema=t.get("inputSchema") or {"type": "object", "properties": {}}))
        return out + CADDY_TOOLS

    # --- judging -----------------------------------------------------------------------------------------
    def judge(self) -> None:
        """Export the design, rebuild it on SMLib and measure everything the spec asks for."""
        g = self._measure()
        if g is None:
            self.graph, self.reports = None, []
        else:
            self._run(g)

    def trial(self, values: dict) -> SizeReport:
        """What the kernel would say if these parameters (caddy ids → numbers) were changed, without keeping them:
        SolidCraft rebuilds inside a checkpoint, SMLib judges the export, and the design is put back."""
        by = {p["id"]: p for p in self.sc_parameters}
        changes = {by[k]["name"]: _plain(dict(by[k], value=float(v))) for k, v in values.items() if k in by}
        keep = dict(facts=dict(self.facts), checks=list(self.spec["checks"]), params=list(self.sc_parameters))
        cp = self._sc_json("execute", {"command": "edit.checkpoint"})["checkpoint"]
        try:
            self._sc("execute", {"command": "parameters.change", "params": {"changes": changes}})
            g = self._measure(drive=False)
            if g is None:
                return SizeReport("image", self.rows[0], None, [], [],
                                  error=dict(block=None, status="BUILD_ERROR", message=self.facts.get("note", "")))
            return build_size(self.ev, g, self.rows[0], self.spec)
        except G.GraphError as e:
            return SizeReport("image", self.rows[0], None, [], [],
                              error=dict(block=None, status="BUILD_ERROR", message=str(e)))
        finally:
            self._sc("execute", {"command": "edit.restore_checkpoint", "params": {"checkpoint": cp}})
            self.facts.clear(); self.facts.update(keep["facts"])
            self.spec["checks"], self.sc_parameters = keep["checks"], keep["params"]
            self.revision = self._revision()

    def _measure(self, drive: bool = True) -> dict | None:
        """The current SolidCraft design as a caddy graph of SMLib solids, with the facts the checks read.
        `drive=False` (a what-if) keeps the last answer to "does every parameter move the geometry?"."""
        f = self.facts
        dead = {k: f.get(k) for k in ("dead_parameters", "dead")}
        f.clear()
        f["undeclared"] = 0 if self.declared else 1
        self.revision = self._revision()
        m = self._sc_json("measure", {"summary": True})
        bodies = [b for b in m.get("bodies", []) if not b.get("error")]
        solids = [b for b in bodies if b.get("closed") and b.get("volume_mm3")]
        if not solids:
            f["note"] = "the design has no closed solid body yet" if not bodies else "no body is a closed solid"
            return None
        f["volume_mm3"] = float((m.get("total") or {}).get("volume_mm3") or sum(b["volume_mm3"] for b in solids))
        f["bodies"] = [b.get("name") for b in bodies]
        f["unmeasured"] = [dict(name=b.get("name"), error=str(b.get("error"))[:160]) for b in m.get("bodies", [])
                           if b.get("error")]
        self._timeline(f)
        self._parameters(f, drive)
        if not drive:
            f.update(dead)
        step = self.dir / "model.step"
        self._sc("export", {"path": str(step), "format": "step"})
        tag = hashlib.sha1(step.read_bytes().split(b"DATA;", 1)[-1]).hexdigest()[:10]   # the geometry, not the header's date
        try:
            stock = judge.import_step(step, self.dir, tag)
            self._prune(set(stock))
        except Exception as e:  # SMLib could not read what SolidCraft wrote: a failure with its reason
            f["note"] = f"SMLib could not read the exported STEP ({type(e).__name__}: {e})"
            return None
        g = self._graph(stock, solids)
        if self.declared and self.declared.get("symmetry_plane") in judge.AXIS:
            self._set_checks(judge.symmetry_tolerance({n: sq_kernel.load_stock(s) for n, s in zip(g["bodies"], stock)}))
        self._picture(f)
        return g

    def _prune(self, new: set[str]) -> None:
        """Drop the SMLib copies of designs that are gone (the judged one and the new one stay)."""
        keep = new | {b["stock"] for b in (self.graph or {}).get("blocks", [])}
        for f in self.dir.glob("sc_*.smb"):
            if f.stem not in keep:
                f.unlink(missing_ok=True)
                sq_kernel._stock_cache.pop(f.stem, None)

    def _graph(self, stock: list[str], solids: list[dict]) -> dict:
        """One stock block per imported solid, named after the SolidCraft body with the closest volume."""
        import usd_brep as u
        left, taken, blocks, bodies = list(solids), set(), [], {}
        for s in stock:
            v = float(u.compute_volume(sq_kernel.load_stock(s)))
            best = min(left, key=lambda b: abs(b["volume_mm3"] - v)) if left else None
            if best is not None:
                left.remove(best)
            name = judge.ident(best.get("name", "body") if best else "body", taken)
            blocks.append(dict(id=name, op="stock", part=name, role="stock", stock=s,
                               aliases={"side": f"{name}.surface", "cap": f"{name}.flat"}))
            bodies[name] = name
        params = [dict(name=p["id"], displayName=p["name"], part=blocks[0]["part"], unit=p.get("unit", ""),
                       default=p["value"]) for p in self.sc_parameters if p.get("id")]
        return dict(family="image", title=self.spec["title"], parameters=params, blocks=blocks, bodies=bodies)

    def _timeline(self, f: dict) -> None:
        d = self._sc_json("inspect_design", {"measure": False, "sketches": False, "joints": False,
                                             "timeline": "summary", "bodies": False})
        self._design = d
        bad = [dict(feature=t.get("name") or t.get("id"), error=t.get("error")) for t in d.get("timeline", [])
               if t.get("error")]
        f["feature_errors"], f["failing_features"] = len(bad), bad
        f["features"] = len(d.get("timeline", []))

    def _parameters(self, f: dict, drive: bool = True) -> None:
        """The user parameters, and for each one whether it rebuilds cleanly and moves the geometry (tried a little
        larger, then a little smaller, through SolidCraft's check_parameters, which puts the design back)."""
        taken: set[str] = set()
        self.sc_parameters, dead, base = [], [], None
        for p in user_parameters(self._design):
            v = p["value"]
            p["id"] = judge.ident(p["name"], taken)
            self.sc_parameters.append(p)
            if not drive:
                continue
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                dead.append(dict(name=p["name"], why="its value is not a number"))
                continue
            if base is None:     # the design as it is, read the same way the trials are
                base = _signature(self._sc_json("check_parameters", {"set": {p["name"]: _plain(p)}}))
            whole = not p.get("unit") and float(v).is_integer()
            tries = [v + 1, v - 1] if whole else [v * 1.07, v * 0.93] if v else [1.0]
            why = "changing it does not change the geometry"
            for t in tries:
                val = f"{t:g} {p['unit']}".strip() if p.get("unit") else t
                try:
                    r = self._sc_json("check_parameters", {"set": {p["name"]: val}})
                except G.GraphError as e:
                    why = f"it does not rebuild at {val}: {str(e)[:160]}"
                    continue
                bad = r.get("errors") or r.get("parameter_errors")
                if bad:
                    why = f"it does not rebuild at {val}: {json.dumps(bad[:1])[:200]}"
                    continue
                if _signature(r) != base:
                    why = None
                    break
            if why:
                dead.append(dict(name=p["name"], why=why))
        f["parameters"], f["dead_parameters"], f["dead"] = len(self.sc_parameters), len(dead), dead

    def _picture(self, f: dict) -> None:
        view = (self.declared or {}).get("picture_view")
        if not view:
            f["picture_note"] = "describe_object has not said how the picture was taken (picture_view)"
            return
        args = {"reference": str(self.reference), **{k: v for k, v in view.items() if v is not None}}
        r = self.mcp.call("compare_picture", args)
        out = r.json() or {}
        if r.is_error or "overlap" not in out:
            f["picture_note"] = f"SolidCraft could not compare the picture: {r.text[:240]}"
            return
        f["overlap"] = round(float(out["overlap"]), 4)
        f["picture"] = {k: out[k] for k in ("aspect_mismatch", "scale", "fit") if k in out}

    # --- what the model reads ----------------------------------------------------------------------------
    def report(self) -> str:
        f = self.facts
        if not self.reports:
            return f"caddy's kernel check: nothing to judge yet ({f.get('note', 'no solid body')})."
        r = self.reports[0]
        if r.error:
            return f"caddy's kernel check: SMLib could not rebuild the exported solids: {json.dumps(r.error)}"
        lines = [f"caddy's kernel check (SMLib, {len(r.result.bodies)} bod{'y' if len(r.result.bodies) == 1 else 'ies'}, "
                 f"{f.get('features', 0)} features, {f.get('parameters', 0)} parameters): "
                 + ("every check ✓" if r.ok else "NOT accepted")]
        for c in r.checks:
            lines.append(f"  {'✓' if c['ok'] else '✗'} {c['id']}: measured {c['measured']} "
                         f"(needs {c['op']} {c['expected']}) · {c.get('rule') or ''}{self._hint(c)}")
        for s in r.smoke:
            if not s["ok"]:
                lines.append(f"  ✗ {s['id']}" + (f": measured {s['measured']}" if "measured" in s else ""))
        return "\n".join(lines)

    def _hint(self, c: dict) -> str:
        f = self.facts
        if c["ok"]:
            return ""
        if c["id"] == "symmetric" and f.get("mirror", {}).get("at"):
            m = f["mirror"]
            return (f"\n      the surface of body {m['body']} near {m['at']} has no mirror partner across {m['plane']} "
                    f"(plane at {m['mid']} mm). Mirror the feature (solid.mirror) or centre its sketch on the plane.")
        if c["id"] == "symmetric":
            return "\n      call describe_object first: say which plane the object mirrors across."
        if c["id"] == "features_healthy":
            return "\n      failing: " + json.dumps(f.get("failing_features", [])[:4])[:400]
        if c["id"] == "parameters_drive":
            return "\n      " + "; ".join(f"{d['name']}: {d['why']}" for d in f.get("dead", [])[:6]) + \
                   ". Use each parameter in a dimension or feature field (an expression such as \"w/2 - t\"), or remove it."
        if c["id"] == "parametric":
            return "\n      add named parameters (parameters.add) for the object's main sizes and build from them."
        if c["id"] == "matches_picture":
            return "\n      " + (f.get("picture_note") or "compare the outlines (screenshot from the picture's view) and "
                                 "fix the proportions that differ, or correct picture_view in describe_object.")
        if c["id"] == "export_intact":
            bad = f.get("unmeasured") or []
            return (f"\n      SolidCraft {f.get('volume_mm3')} mm³ vs SMLib {round(f.get('smlib_volume_mm3', 0), 2)} mm³. "
                    + ("SolidCraft could not measure " + "; ".join(f"{b['name']} ({b['error']})" for b in bad[:4])
                       + ": rebuild that body another way (e.g. its own mirror or box feature)." if bad else
                       "A body did not survive the STEP export as a closed solid."))
        return ""

    def family_table(self) -> str:
        return self.report()

    # --- tools -------------------------------------------------------------------------------------------
    def describe_object(self, a: dict) -> str:
        plane = a.get("symmetry_plane")
        if plane not in (*judge.AXIS, "none"):
            raise G.GraphError('symmetry_plane must be "YZ", "XZ", "XY" or "none"')
        if plane == "none" and len((a.get("asymmetry_evidence") or "").strip()) < 12:
            raise G.GraphError("an object is assumed mirror-symmetric unless the picture shows otherwise: to declare "
                               '"none", say in asymmetry_evidence what in the picture shows it')
        view = a.get("picture_view") or {}
        if not (view.get("view") or view.get("camera")):
            raise G.GraphError("picture_view needs a view (front, right, top, iso, …) or a camera")
        self.declared = dict(object=a.get("object", ""), symmetry_plane=plane,
                             asymmetry_evidence=a.get("asymmetry_evidence", ""), picture_view=view,
                             assumptions=a.get("assumptions", ""))
        self._set_checks()
        if self.revision is not None and self._revision() == self.revision and self.graph is not None:
            self.judge()
        return "noted.\n" + self.report()

    def call(self, name: str, args: dict) -> tuple[str, list[tuple[str, str]], bool]:
        """Run one tool. Returns (text for the model, images, is_error)."""
        if name == "describe_object":
            return self.describe_object(args), [], False
        if name == "compare_picture":
            return self.compare_picture(args)
        if name == "finish":
            if self._revision() != self.revision or not self.reports:
                self.judge()
            if self.ok:
                self.save()
                return ("accepted: the kernel confirms every check ✓\nsaved " +
                        ", ".join(str(p) for p in self.saved.values())), [], False
            return "not accepted: the kernel still reports failures\n" + self.report(), [], True
        if name not in PASS_THROUGH or name not in self._tools:
            return f"unknown tool {name}", [], True
        r = self.mcp.call(name, args)
        text = r.text if len(r.text) <= MAX_TOOL_TEXT else (
            r.text[:MAX_TOOL_TEXT] + f"\n… ({len(r.text) - MAX_TOOL_TEXT} more characters cut: ask for less, e.g. "
                                     f"verbose:false, summary:true, or a filter)")
        if name in CHANGES_DESIGN and self._revision() != self.revision:
            self.judge()                  # a failed batch can still have changed the design (rollback:false)
            text += "\n\n" + self.report()
        return text, r.images, r.is_error

    def compare_picture(self, a: dict) -> tuple[str, list, bool]:
        """SolidCraft's outline comparison against the uploaded picture (the path is caddy's, never the model's)."""
        args = {k: v for k, v in a.items() if k != "reference"}
        r = self.mcp.call("compare_picture", {"reference": str(self.reference), **args})
        return r.text, r.images, r.is_error

    def save(self) -> dict:
        """Write the accepted design next to the picture: SolidCraft's `.sc` (parametric) and the STEP SMLib judged."""
        sc = self.dir / "model.sc"
        self._sc("save", {"path": str(sc)})
        self.saved = dict(design=sc, step=self.dir / "model.step")
        return self.saved

    def set_parameter(self, name: str, value) -> str:
        """A value change from the board: SolidCraft rebuilds the timeline, SMLib judges the result. No model call."""
        self._sc("set_parameter", {"name": name, "value": value})
        self.judge()
        return self.report()

    def set_values(self, values: dict) -> str:
        """Several at once, by caddy id, as the Feature tab applies them (one rebuild in SolidCraft)."""
        by = {p["id"]: p for p in self.sc_parameters}
        changes = {by[k]["name"]: _plain(dict(by[k], value=float(v))) for k, v in values.items()}
        self._sc("execute", {"command": "parameters.change", "params": {"changes": changes}})
        self.judge()
        if self.saved:          # the files an accepted run wrote follow the design
            self.save()
        return self.report()

    def screenshot(self, view: str = "iso", width: int = 480, height: int = 320) -> str | None:
        r = self.mcp.call("screenshot", {"view": view, "width": width, "height": height, "source": "model"})
        return None if r.is_error or not r.images else f"data:{r.images[0][0]};base64,{r.images[0][1]}"

    def image_block(self) -> dict:
        return {"type": "image", "source": {"type": "base64", "media_type": self.mime,
                                            "data": base64.standard_b64encode(self.image).decode()}}

    def verdict(self) -> dict:
        v = super().verdict()
        if not self.reports:
            v["failing_checks"] = ["build"]
        return v


def _plain(p: dict):
    return f"{p['value']:g} {p['unit']}".strip() if p.get("unit") else p["value"]


def _signature(r: dict) -> str:
    """What a parameter trial produced, for comparing two trials: sizes, centres of mass and every cylinder axis."""
    keep = {k: r.get(k) for k in ("measure", "cylinders")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


def user_parameters(design: dict) -> list[dict]:
    """The named parameters of a SolidCraft design as [{name, value, unit, expression}]."""
    out = []
    for p in design.get("params") or []:
        if not isinstance(p, dict) or not p.get("name") or p.get("model"):   # model: a feature's own dimension
            continue
        out.append(dict(name=p["name"], value=p.get("value"), unit=p.get("unit") or "",
                        expression=p.get("expression", "")))
    return out


CADDY_TOOLS = [
    {"name": "describe_object",
     "description": "Tell caddy what the picture shows before you build: the object, the plane it mirrors across, "
                    "and how the picture was taken. caddy then checks the model's symmetry on its own kernel and "
                    "compares the model's outline with the picture from that view. Call it again to correct it.",
     "input_schema": {"type": "object", "properties": {
         "object": {"type": "string", "description": "what it is, e.g. 'four-legged dining chair with a slatted back'"},
         "symmetry_plane": {"type": "string", "enum": ["YZ", "XZ", "XY", "none"],
                            "description": "the world plane the model mirrors across. Build upright (Z up), front "
                                           "towards -Y, centred on the origin: left/right symmetry is then YZ. "
                                           "'none' only when the picture shows the object is not symmetric."},
         "asymmetry_evidence": {"type": "string", "description": "required with 'none': what in the picture shows it"},
         "picture_view": {"type": "object", "description": "how the picture sees the model, as compare_picture takes it",
                          "properties": {
                              "view": {"type": "string", "description": "front, back, left, right, top, bottom or iso"},
                              "camera": {"type": "object", "description": "{target, yaw, pitch, distance, fov?} in mm and radians"},
                              "crop_fraction": {"type": "array", "items": {"type": "number"},
                                                "description": "[x, y, w, h] as fractions 0..1: the part of the picture that shows the object"},
                              "fit": {"type": "string", "enum": ["width", "height", "both"]}}},
         "assumptions": {"type": "string", "description": "sizes and hidden features you took from knowledge of this "
                                                          "kind of object, not from the picture"}},
         "required": ["object", "symmetry_plane", "picture_view"]}},
    {"name": "compare_picture",
     "description": "Compare the model's outline with the object in the user's picture (SolidCraft's compare_picture; "
                    "caddy supplies the picture). Returns the overlap (0..1), where the model's edges sit against the "
                    "picture's, and an overlay image (model red, picture blue). Use it to find the view and fix "
                    "proportions.",
     "input_schema": {"type": "object", "properties": {
         "view": {"type": "string"}, "camera": {"type": "object"},
         "crop_fraction": {"type": "array", "items": {"type": "number"}},
         "crop_px": {"type": "array", "items": {"type": "number"}}, "part_fills_crop": {"type": "boolean"},
         "fit": {"type": "string", "enum": ["width", "height", "both"]},
         "bodies": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "finish",
     "description": "Declare the model done. Refused (with the failing checks) unless caddy's kernel accepts it; when "
                    "accepted, caddy saves the SolidCraft design (.sc) and the STEP file.",
     "input_schema": {"type": "object", "properties": {}}},
]
