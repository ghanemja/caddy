"""Image → parametric CAD: Claude looks at a picture and models it in SolidCraft; caddy's kernel decides.

The loop is caddy's own (caddy/agent/loop.py) with two differences: the task carries the picture, and tool results
can carry pictures back (SolidCraft renders and outline overlays), so Claude can compare what it built with what it
was shown.
"""
from __future__ import annotations

import json
import time

from caddy.agent.loop import MODEL, Agent, RunResult
from caddy.kernel.build import KernelError
from caddy.kernel.graph import GraphError

from .mcp import McpError
from .session import ImageSession

SYSTEM = """You turn a picture of an object into an editable, parametric CAD model. You model in SolidCraft (a \
parametric CAD program with sketches, constraints and a feature timeline) through its tools. caddy then exports your \
design, rebuilds it on a second geometry kernel (NVIDIA SMLib) and measures it. You never judge whether the model is \
good enough: caddy's kernel checks do, and `finish` is refused while one fails.

WHAT A GOOD RESULT IS
An engineer opens your design in SolidCraft, changes `seat_height` and the whole object follows. So:
- every main size is a named parameter (parameters.add) with a unit, and sketches and features use those names in \
expressions ("seat_w/2 - leg_t/2"), never a bare number that repeats a parameter's value;
- features are named for what they are ("Seat", "FrontLeg", "BackSlat");
- there is no parameter that nothing uses.

USE WHAT YOU KNOW ABOUT THE OBJECT
A single picture never shows everything. Decide what the object is, then fill in what the picture hides from what \
such objects are like, and say so in describe_object's `assumptions`.
- SYMMETRY IS THE DEFAULT. Chairs, tables, couches, aircraft, cars, bottles, tools, brackets: made objects are \
mirror-symmetric left to right (often front to back too) unless the picture shows otherwise. Assume the hidden side \
mirrors the visible one. Build it that way: centre the object on the origin, draw centred sketches or build one side \
and `solid.mirror` it across the symmetry plane, and repeat equal parts (legs, slats, wheels, holes) with patterns. \
Declare "none" only for something the picture actually shows (one armrest, a handle on one side), and say what.
- Equal parts are equal: four legs share one length and section; a table top is level; seats are flat or gently \
contoured; shelves are parallel; wheels share a diameter. One parameter drives all of them.
- Use real proportions and sizes when the picture gives no scale: a dining chair seat is about 450 mm high and 420 \
mm wide, a dining table about 750 mm high, a couch seat about 420 mm high and 550 mm deep, a door handle sits near \
1000 mm, sheet steel is 1–3 mm, a moulded plastic wall 2–3 mm. If the user gives sizes, theirs win.
- Things must be buildable and must stand: parts that touch overlap or share a face (no floating bodies, no gaps), \
legs reach the floor at z = 0, wall thicknesses are plausible.
- Model the shape, not the image: ignore the background, shadows, texture, colour, labels and lens distortion. \
Leave out detail smaller than about 2 % of the object unless it defines what the object is.

FRAME
Z is up and the object stands on z = 0. Its front faces -Y (SolidCraft's front view looks along +Y with +X to the \
right). Centre it so its left/right mirror plane is the world YZ plane (x = 0).

HOW TO WORK
1. Read the modelling reference once: skill_reference {"skill":"solidcraft-modeling"}. It has the command shapes.
2. Look at the picture. Call describe_object: what it is, its symmetry plane, the view the picture was taken from \
(a named view, or a camera once you can estimate one), and your assumptions.
3. Plan in numbers: the parameters, each part with its position, the expected bounding box.
4. Build in a few `batch` calls (parameters and the main body first, then the repeated and mirrored parts, then \
cuts, holes and edge treatments). Pass "verbose": false. After each change caddy appends its kernel check to the \
tool result: read it.
5. Compare with the picture: `screenshot` from the picture's view, then `compare_picture` with the same view. Fix \
the proportions the overlay shows are off by changing parameters (set_parameter), not by redrawing. If the picture \
is at an angle, estimate a camera (yaw, pitch) and refine it; update describe_object's picture_view to the view \
that matches best.
6. Fix whatever caddy's check marks ✗ with small edits (timeline.edit, set_parameter, one more feature). Do not \
start over.
7. Call finish when every check is ✓. Keep your messages short: one or two sentences on what you are doing and why."""


def task(hint: str = "") -> str:
    return ("Here is the picture. Model the object it shows as a parametric SolidCraft design and finish when caddy's "
            "kernel accepts it." + (f"\n\nFrom the user: {hint}" if hint else ""))


class ImageAgent(Agent):
    def __init__(self, session: ImageSession, model: str = MODEL, effort: str = "medium", max_rounds: int = 40,
                 on_event=None, run_id: str | None = None):
        super().__init__(session, model=model, effort=effort, max_rounds=max_rounds, on_event=on_event,
                         run_id=run_id, system=SYSTEM, tools=session.tools(), lang="solidcraft")

    def _call(self, name: str, args: dict) -> tuple[str, list, bool]:
        try:
            return self.s.call(name, args)
        except KernelError as e:
            return f"kernel error: {json.dumps(e.facts())}", [], True
        except GraphError as e:
            return str(e), [], True
        except (KeyError, TypeError, ValueError) as e:
            return f"bad tool input: {type(e).__name__}: {e}", [], True

    def run(self, task_text: str, with_picture: bool = True) -> RunResult:
        s: ImageSession = self.s
        t0 = time.time()
        first = ([s.image_block()] if with_picture else []) + [{"type": "text", "text": task_text}]
        messages: list = [{"role": "user", "content": first}]
        claimed, false_done, rounds, calls = False, 0, 0, 0
        self._emit(dict(type="start", run_id=self.run_id, model=self.model, effort=self.effort, family=s.family,
                        lang=self.lang, task=task_text, picture=s.reference.name, hint=s.hint))
        try:
            while rounds < self.max_rounds:
                resp = self._request(messages)
                rounds += 1
                self.usage.add(resp.usage)
                text = " ".join(b.text for b in resp.content if b.type == "text").strip()
                self._emit(dict(type="response", round=rounds, stop=resp.stop_reason, text=text,
                                usage=resp.usage.model_dump(exclude_none=True)))
                messages.append({"role": "assistant", "content": resp.content})
                uses = [b for b in resp.content if b.type == "tool_use"]
                if resp.stop_reason == "refusal" or not uses:
                    break
                results, accepted = [], False
                for b in uses:
                    calls += 1
                    out, images, is_err = self._call(b.name, dict(b.input))
                    if b.name == "finish":
                        claimed = True
                        false_done += int(is_err)
                        accepted = not is_err
                    self._emit(dict(type="tool", round=rounds, name=b.name, input=b.input, error=is_err, result=out,
                                    images=len(images), kernel=s.verdict(), explain=s.explain(),
                                    **({"declared": s.declared} if b.name == "describe_object" and not is_err else {}),
                                    **({"overlay": f"data:{images[0][0]};base64,{images[0][1]}"}
                                       if b.name == "compare_picture" and images else {}),
                                    **({"saved": {k: str(v) for k, v in s.saved.items()}} if accepted else {})))
                    content = [{"type": "text", "text": out or "(no text)"}] + [
                        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
                        for mime, data in images]
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": content,
                                    **({"is_error": True} if is_err else {})})
                messages.append({"role": "user", "content": results})
                if accepted:
                    break
        except McpError as e:      # SolidCraft went away: the run ends with the reason, not a traceback
            self._emit(dict(type="error", message=f"SolidCraft: {e}"))
        v = s.verdict()
        res = RunResult(self.run_id, v["ok"], claimed, false_done, rounds, calls, self.usage.to_dict(self.model),
                        round(time.time() - t0, 1), self.log_path, v)
        self._emit(dict(type="end", ok=res.ok, claimed_done=claimed, false_done=false_done, rounds=rounds,
                        tool_calls=calls, usage=res.usage, seconds=res.seconds, verdict=v))
        return res
