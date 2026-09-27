"""
Parrot Mambo sensor reader for the laptop supervisor.
Drone connected via USB at 192.168.2.1.
Checks if /tmp/dragon.log exists first. If yes, tails it directly.
If no, kills dragon-prog and restarts with log redirect.
"""
import re, telnetlib, threading, time

_DRONE_IP = "192.168.2.1"
_SENSOR_RE = re.compile(
    r"sonar altitude: ([\d.]+)meters.*?"
    r"rsedu_control\(\): accel x: ([-\d.]+)m/s2, y: ([-\d.]+)m/s2, z: ([-\d.]+)m/s2",
    re.DOTALL,
)
_lock    = threading.Lock()
_latest  = {"sonar_m": 0.0, "accel_mag": 0.0}
_started = False


def _loop():
    while True:
        try:
            t = telnetlib.Telnet(_DRONE_IP, 23, timeout=5)
            time.sleep(0.5)
            t.read_very_eager()

            # Check if log already exists
            t.write(b"ls /tmp/dragon.log 2>/dev/null && echo EXISTS || echo MISSING\n")
            time.sleep(0.5)
            status = t.read_very_eager().decode(errors="replace")

            if "MISSING" in status:
                # Kill and restart with log redirect
                t.write(b"kill $(ps | grep dragon-prog | grep -v grep | awk '{print $1}' | head -1) 2>/dev/null; sleep 1; /usr/bin/dragon-prog > /tmp/dragon.log 2>&1 &\n")
                time.sleep(6)
                t.read_very_eager()

            t.write(b"tail -f /tmp/dragon.log\n")
            buf = ""
            while True:
                chunk = t.read_very_eager()
                if chunk:
                    buf += chunk.decode(errors="replace")
                    if len(buf) > 8192:
                        buf = buf[-8192:]
                    m = _SENSOR_RE.search(buf)
                    if m:
                        sonar = float(m.group(1))
                        ax, ay, az = float(m.group(2)), float(m.group(3)), float(m.group(4))
                        with _lock:
                            _latest["sonar_m"]   = round(sonar, 4)
                            _latest["accel_mag"] = round((ax**2+ay**2+az**2)**0.5, 4)
                        buf = buf[buf.rfind("accel"):]
                else:
                    time.sleep(0.1)
        except Exception:
            time.sleep(3)


def start():
    global _started
    if not _started:
        _started = True
        threading.Thread(target=_loop, daemon=True).start()


def read() -> dict:
    with _lock:
        return dict(_latest)
