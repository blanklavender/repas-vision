"""Host-side driver for the Flora Inspection Station turntable.

Pairs with the serial layer added to Noah's sketch. Motion calls block until
the Arduino reports the move finished, so a capture loop can just alternate
move / capture with no sleeps.

    pip install pyserial

    from turntable import Turntable

    with Turntable("/dev/ttyACM0") as tt:      # "COM5" on Windows
        tt.set_speed(5)
        tt.home()
        for angle in range(0, 360, 15):
            tt.goto(angle)
            capture_frame(angle)               # your RealSense / Femto Bolt grab

Angles are whole degrees -- the firmware's moveBigGear() takes an int.
"""

import re
import time

import serial

_STATUS = re.compile(r"^OK\s+ANG:(-?\d+)\s+SPD:(\d+)\s+INCR:(\d+)")


class TurntableError(RuntimeError):
    pass


class Turntable:
    def __init__(self, port, baud=9600, boot_timeout=90.0, timeout=180.0):
        self.timeout = timeout
        # Opening the port pulses DTR and resets the Arduino, which re-runs the
        # homing routine. Wait for READY rather than guessing at a sleep.
        self.ser = serial.Serial(port, baud, timeout=1.0)
        self.angle = 0
        self.speed = 1
        self.incr = 5
        self._wait_for("READY", boot_timeout)

    # -- lifecycle -------------------------------------------------------
    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- commands --------------------------------------------------------
    def home(self):
        return self._cmd("H")

    def move(self, degrees):
        """Relative move, whole degrees. Positive = forward."""
        degrees = int(round(degrees))
        if degrees == 0:
            return self.angle
        return self._cmd(f"M{degrees}")

    def goto(self, degrees):
        """Absolute angle 0-359, shorter direction."""
        return self._cmd(f"G{int(round(degrees)) % 360}")

    def set_speed(self, level):
        if not 1 <= int(level) <= 9:
            raise ValueError("speed level must be 1-9")
        return self._cmd(f"S{int(level)}")

    def set_increment(self, degrees):
        """Set the increment the IR remote's forward/back buttons use."""
        return self._cmd(f"I{int(round(degrees))}")

    def status(self):
        return self._cmd("?")

    # -- plumbing --------------------------------------------------------
    def _cmd(self, text):
        self.ser.reset_input_buffer()
        self.ser.write((text + "\n").encode())
        self.ser.flush()
        return self._read_status()

    def _read_status(self):
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="replace").strip()
            if not line:
                continue
            if line.startswith("ERR"):
                raise TurntableError(line)
            m = _STATUS.match(line)
            if m:
                ang, spd, incr = m.groups()
                self.angle, self.speed, self.incr = int(ang), int(spd), int(incr)
                return self.angle
        raise TurntableError("timed out waiting for the turntable")

    def _wait_for(self, token, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="replace").strip()
            if line == token:
                return
        raise TurntableError(f"never saw {token} - wrong port, or homing stalled")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Turntable smoke test")
    p.add_argument("port", help="/dev/ttyACM0, /dev/ttyUSB0, COM5 ...")
    p.add_argument("--step", type=int, default=45)
    p.add_argument("--speed", type=int, default=5)
    args = p.parse_args()

    with Turntable(args.port) as tt:
        tt.set_speed(args.speed)
        tt.home()
        for _ in range(360 // args.step):
            print(f"at {tt.move(args.step):4d} deg")