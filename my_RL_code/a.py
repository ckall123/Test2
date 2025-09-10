import re
import time
import json
import math
import shutil
import select
import subprocess
from typing import Optional, Callable, List

class ContactEvent:
    def __init__(self, event: str, pair: str, force_z: Optional[float]):
        self.event = event
        self.pair = pair
        self.force_z = force_z
        self.timestamp = time.time()

    def to_dict(self):
        data = {"event": self.event, "pair": self.pair}
        if self.force_z is not None:
            data["fz"] = self.force_z
        return data

class ContactEventTracker:
    def __init__(self,
                 topic: str = "/gazebo/default/physics/contacts",
                 name_filter: str = r"(UF_ROBOT|beer|table|bowl)",
                 pair_key_mode: str = "link",
                 min_update_gap: float = 1.0,
                 inactive_end: float = 2.0,
                 restart_grace: float = 1.0,
                 force_delta_abs: float = 1.0,
                 force_delta_pct: float = 0.30,
                 force_abs_max: float = 1e6,
                 min_start_depth: float = 5e-5,
                 peak_only: bool = True,
                 settle_suppress: float = 0.2,
                 stall_timeout: float = 3.0,
                 reconnect_delay: float = 0.2,
                 on_event: Optional[Callable[[ContactEvent], None]] = None):

        self.topic = topic
        self.name_filter = re.compile(name_filter)
        self.pair_key_mode = pair_key_mode

        self.min_update_gap = min_update_gap
        self.inactive_end = inactive_end
        self.restart_grace = restart_grace
        self.force_delta_abs = force_delta_abs
        self.force_delta_pct = force_delta_pct
        self.force_abs_max = force_abs_max
        self.min_start_depth = min_start_depth
        self.peak_only = peak_only
        self.settle_suppress = settle_suppress
        self.stall_timeout = stall_timeout
        self.reconnect_delay = reconnect_delay

        self.on_event = on_event
        self._active = {}
        self._grace = {}
        self._buffered_events: List[ContactEvent] = []

        self.re_start = re.compile(r'^\s*contact\s*{')
        self.re_names = re.compile(r'collision[12]:\s*"([^"]+)"')
        self.re_fz = re.compile(r'body_1_wrench\s*{[^}]*force\s*{[^}]*z:\s*([-+0-9.eE]+)', re.S)
        self.re_depth = re.compile(r'^\s*depth:\s*([-+0-9.eE]+)', re.M)

    class _State:
        def __init__(self, t: float, fz: Optional[float]):
            self.last_seen = t
            self.last_emit = 0.0
            self.last_fz = fz
            self.peak_abs = abs(fz) if fz is not None else 0.0
            self.start_time = t

    def _gz_bin(self) -> str:
        return shutil.which("gz") or shutil.which("ign") or "gz"

    def _key_from(self, full: str) -> str:
        parts = full.split("::")
        if self.pair_key_mode == "model":
            return parts[0]
        elif self.pair_key_mode == "link" and len(parts) >= 2:
            return f"{parts[0]}::{parts[1]}"
        return full

    def _canon_pair(self, a: str, b: str) -> str:
        a, b = self._key_from(a), self._key_from(b)
        return " <-> ".join(sorted([a, b]))

    def _parse_pair(self, block: str) -> Optional[str]:
        names = self.re_names.findall(block)
        return self._canon_pair(names[0], names[1]) if len(names) >= 2 else None

    def _parse_fz(self, block: str) -> Optional[float]:
        m = self.re_fz.search(block)
        if not m:
            return None
        try:
            v = float(m.group(1))
            return v if math.isfinite(v) and abs(v) < self.force_abs_max else None
        except Exception:
            return None

    def _parse_max_depth(self, block: str) -> Optional[float]:
        vals = []
        for s in self.re_depth.findall(block):
            try:
                v = float(s)
                if math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
        return max(vals) if vals else None

    def _should_emit_update(self, state: '_State', fz: Optional[float], now: float) -> bool:
        if fz is None:
            return False
        if self.settle_suppress > 0 and (now - state.start_time) < self.settle_suppress:
            return False
        if (now - state.last_emit) < self.min_update_gap:
            return False

        absfz = abs(fz)

        if self.peak_only:
            higher_abs = absfz > state.peak_abs + self.force_delta_abs
            higher_pct = absfz > state.peak_abs * (1.0 + self.force_delta_pct)
            return higher_abs or higher_pct
        else:
            if state.last_fz is None:
                return True
            delta = abs(fz - state.last_fz)
            rel_ok = (abs(state.last_fz) > 0.0) and (delta >= abs(state.last_fz) * self.force_delta_pct)
            return (delta >= self.force_delta_abs) or rel_ok

    def _emit(self, event: str, pair: str, fz: Optional[float]):
        evt = ContactEvent(event, pair, fz)
        self._buffered_events.append(evt)
        if self.on_event:
            self.on_event(evt)
        else:
            print(json.dumps(evt.to_dict()), flush=True)

    def poll_events(self) -> List[ContactEvent]:
        events = self._buffered_events
        self._buffered_events = []
        return events

    def start(self):
        while True:
            proc = None
            try:
                proc = subprocess.Popen(
                    [self._gz_bin(), "topic", "-u", "-e", self.topic],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, bufsize=1
                )
                last_byte = time.time()
                buf, brace = [], 0

                while True:
                    r, _, _ = select.select([proc.stdout], [], [], 0.5)
                    if r:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        last_byte = time.time()

                        if not buf:
                            if self.re_start.search(line):
                                buf = [line]
                                brace = line.count("{") - line.count("}")
                        else:
                            buf.append(line)
                            brace += line.count("{") - line.count("}")
                            if brace <= 0:
                                block = "".join(buf)
                                buf, brace = [], 0

                                pair = self._parse_pair(block)
                                if not pair or not self.name_filter.search(pair):
                                    continue

                                now = time.time()
                                fz = self._parse_fz(block)
                                depth = self._parse_max_depth(block)
                                state = self._active.get(pair)

                                if state is None:
                                    if depth is not None and depth < self.min_start_depth:
                                        continue
                                    if now < self._grace.get(pair, 0.0):
                                        state = self._State(now, fz)
                                        self._emit("update", pair, fz)
                                    else:
                                        state = self._State(now, fz)
                                        self._emit("start", pair, fz)
                                    state.last_emit = now
                                    self._active[pair] = state
                                else:
                                    state.last_seen = now
                                    if self._should_emit_update(state, fz, now):
                                        self._emit("update", pair, fz)
                                        state.last_emit = now
                                        if fz is not None:
                                            state.peak_abs = max(state.peak_abs, abs(fz))
                                    state.last_fz = fz

                    now = time.time()
                    expired = [k for k, v in self._active.items() if now - v.last_seen >= self.inactive_end]
                    for key in expired:
                        self._emit("end", key, None)
                        self._grace[key] = now + self.restart_grace
                        del self._active[key]

                    if time.time() - last_byte > self.stall_timeout:
                        raise TimeoutError("gz echo stalled")

            except Exception:
                pass
            finally:
                if proc and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1.0)
            time.sleep(self.reconnect_delay)

if __name__ == "__main__":
    def simple_logger(event: ContactEvent):
        print(f"[{event.event}] {event.pair} | fz={event.force_z:.3f}" if event.force_z else f"[{event.event}] {event.pair}")

    tracker = ContactEventTracker(on_event=simple_logger)
    tracker.start()