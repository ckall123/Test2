#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal contacts tailer for Gazebo Classic (peak-only updates)
- Input: `gz topic -u -e /gazebo/.../physics/contacts`
- Parses single `contact { ... }` blocks (brace-matched)
- Emits compact JSON events: start / update / end
- Noise control:
    * UPDATE_MIN_GAP:   min time between updates for same pair
    * FORCE_DELTA_ABS:  absolute delta threshold
    * FORCE_DELTA_PCT:  relative delta threshold
    * PEAK_ONLY_UPDATES: only emit when a new |fz| peak is reached
    * SETTLE_SUPPRESS:  suppress updates for a short time after start
"""

import re, os, sys, time, json, math, shutil, select, subprocess
from typing import Optional

# -------------------- Config (tweak here) --------------------
TOPIC = "/gazebo/default/physics/contacts"
NAME_ALLOW = r"(UF_ROBOT|beer|table|bowl)"  # r".*" to allow all
PAIR_KEY_MODE = "link"                      # collision | link | model

UPDATE_MIN_GAP   = 1.0   # s, min gap between updates per pair
INACTIVE_END     = 2.0   # s, no sight -> emit end
RESTART_GRACE    = 1.0   # s, after end: treat quick reappearance as continuation

FORCE_DELTA_ABS  = 1.0   # N, |Δfz| needed for update (absolute)
FORCE_ABS_MAX    = 1e6   # N, ignore absurd values
FORCE_DELTA_PCT  = 0.30  # 30% relative change threshold (used with peak & non-peak modes)
MIN_START_DEPTH  = 5e-5  # m, require at least this depth if depth is present

# Peak-only + settle suppression
PEAK_ONLY_UPDATES = True  # only report new |fz| peaks inside a contact burst
SETTLE_SUPPRESS   = 0.20  # s, suppress updates shortly after "start" to avoid initial jitter

STALL_TIMEOUT    = 3.0    # s, no bytes -> restart echo
RECONNECT_DELAY  = 0.2    # s, wait before reconnect
# -------------------------------------------------------------

# Regex for unformatted (-u) single-contact blocks
RE_START   = re.compile(r'^\s*contact\s*{')
RE_NAMES   = re.compile(r'collision[12]:\s*"([^"]+)"')
RE_FZ      = re.compile(r'body_1_wrench\s*{[^}]*force\s*{[^}]*z:\s*([-+0-9.eE]+)', re.S)
RE_DEPTH   = re.compile(r'^\s*depth:\s*([-+0-9.eE]+)', re.M)

def gz_bin() -> str:
    return shutil.which("gz") or shutil.which("ign") or "gz"

def key_from(full: str) -> str:
    parts = full.split("::")
    if PAIR_KEY_MODE == "model":    return parts[0]
    if PAIR_KEY_MODE == "link" and len(parts) >= 2: return f"{parts[0]}::{parts[1]}"
    return full  # collision or fallback

def canon_pair(a: str, b: str) -> str:
    a, b = key_from(a), key_from(b)
    return " <-> ".join(sorted([a, b]))

def parse_pair(block: str) -> Optional[str]:
    names = RE_NAMES.findall(block)
    return canon_pair(names[0], names[1]) if len(names) >= 2 else None

def parse_fz(block: str) -> Optional[float]:
    m = RE_FZ.search(block)
    if not m: return None
    try:
        v = float(m.group(1))
        return v if math.isfinite(v) and abs(v) < FORCE_ABS_MAX else None
    except Exception:
        return None

def parse_max_depth(block: str) -> Optional[float]:
    vals = []
    for s in RE_DEPTH.findall(block):
        try:
            v = float(s)
            if math.isfinite(v): vals.append(v)
        except Exception:
            pass
    return max(vals) if vals else None

class State:
    __slots__ = ("last_seen", "last_emit", "last_fz", "peak_abs", "start_time")
    def __init__(self, t: float, fz: Optional[float]):
        self.last_seen = t
        self.last_emit = 0.0
        self.last_fz   = fz
        self.peak_abs  = abs(fz) if fz is not None else 0.0
        self.start_time = t

def emit(event: str, pair: str, fz: Optional[float]):
    rec = {"event": event, "topic": TOPIC, "pair": pair}
    if fz is not None: rec["fz"] = fz
    print(json.dumps(rec), flush=True)

def should_emit_update(st: State, fz: Optional[float], now: float) -> bool:
    """Decide if we should emit an 'update' given current state and force."""
    if fz is None:
        return False
    # settle suppression
    if SETTLE_SUPPRESS > 0 and (now - st.start_time) < SETTLE_SUPPRESS:
        return False
    # min gap
    if (now - st.last_emit) < UPDATE_MIN_GAP:
        return False

    absfz = abs(fz)

    if PEAK_ONLY_UPDATES:
        # emit only when a new |fz| peak is reached by enough margin
        higher_abs = absfz > st.peak_abs + FORCE_DELTA_ABS
        higher_pct = absfz > st.peak_abs * (1.0 + FORCE_DELTA_PCT)
        return higher_abs or higher_pct
    else:
        # original absolute threshold + relative threshold vs last_fz
        if st.last_fz is None:
            return True
        delta = abs(fz - st.last_fz)
        rel_ok = (abs(st.last_fz) > 0.0) and (delta >= abs(st.last_fz) * FORCE_DELTA_PCT)
        return (delta >= FORCE_DELTA_ABS) or rel_ok

def tail():
    gz = gz_bin()
    active, grace = {}, {}  # pair -> State / grace-deadline

    while True:
        proc = None
        try:
            proc = subprocess.Popen(
                [gz, "topic", "-u", "-e", TOPIC],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1
            )
            last_byte = time.time()
            buf, brace = [], 0

            while True:
                r, _, _ = select.select([proc.stdout], [], [], 0.5)
                if r:
                    line = proc.stdout.readline()
                    if not line: break
                    last_byte = time.time()

                    if not buf:
                        if RE_START.search(line):
                            buf = [line]
                            brace = line.count("{") - line.count("}")
                    else:
                        buf.append(line)
                        brace += line.count("{") - line.count("}")
                        if brace <= 0:
                            block = "".join(buf)
                            buf, brace = [], 0

                            pair = parse_pair(block)
                            if not pair or not re.search(NAME_ALLOW, pair):
                                pass
                            else:
                                now = time.time()
                                fz = parse_fz(block)
                                depth = parse_max_depth(block)
                                st = active.get(pair)

                                if st is None:
                                    # optional start gate by depth (only if depth present)
                                    if depth is not None and depth < MIN_START_DEPTH:
                                        pass
                                    else:
                                        # continuation within grace -> treat as update, else -> start
                                        if now < grace.get(pair, 0.0):
                                            st = State(now, fz); active[pair] = st
                                            emit("update", pair, fz); st.last_emit = now
                                            if fz is not None: st.peak_abs = abs(fz)
                                        else:
                                            st = State(now, fz); active[pair] = st
                                            emit("start", pair, fz);  st.last_emit = now
                                            if fz is not None: st.peak_abs = abs(fz)
                                else:
                                    st.last_seen = now
                                    if should_emit_update(st, fz, now):
                                        emit("update", pair, fz); st.last_emit = now
                                        if fz is not None:
                                            st.peak_abs = max(st.peak_abs, abs(fz))
                                    st.last_fz = fz

                # end detection
                now = time.time()
                ended = [p for p, st in active.items() if now - st.last_seen >= INACTIVE_END]
                for p in ended:
                    emit("end", p, None)
                    # store grace deadline and drop state
                    del active[p]
                    grace[p] = now + RESTART_GRACE

                # stall watchdog
                if time.time() - last_byte > STALL_TIMEOUT:
                    raise TimeoutError("echo stalled")

        except Exception:
            # swallow and reconnect
            pass
        finally:
            if proc:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            proc.kill(); proc.wait(timeout=1.0)
                except Exception:
                    pass
        time.sleep(RECONNECT_DELAY)

if __name__ == "__main__":
    tail()
