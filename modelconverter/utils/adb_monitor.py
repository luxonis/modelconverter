from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import threading
import time
import statistics
from collections.abc import Sequence
from loguru import logger
import depthai as dai

@dataclass
class _BaseAdbMonitor(ABC):
    """
    Base class for simple threaded ADB monitors.

    Subclasses implement `_read()` to return a single sample value (or None).
    The base class handles starting/stopping a background sampling thread and
    storing collected measurements.
    """

    adb_handler: object
    interval: float = 0.5

    _measurements: list[object] | None = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    @abstractmethod
    def _read(self) -> object | None:
        """
        Perform a single sampling operation.

        Returns
        -------
        object | None
            A single measurement value (type defined by subclass), or None
            if the sample could not be read.
        """
        ...

    @property
    def measurements(self) -> Sequence[object]:
        """
        Sequence of all successfully collected measurements.

        Returns
        -------
        Sequence[object]
            A list-like view of all values that `_read()` returned
            (excluding None). Empty if the monitor has not run or if
            all reads failed.
        """
        return self._measurements or []

    def start(self) -> None:
        """
        Start the background sampling thread.

        If the monitor is already running, this is a no-op.
        """
        if self._running:
            return
        self._measurements = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop the background sampling thread and wait for it to finish.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join()

    def _loop(self) -> None:
        """
        Internal sampling loop executed in the background thread.
        """
        assert self._measurements is not None
        while self._running:
            try:
                val = self._read()
                if val is not None:
                    self._measurements.append(val)
            except Exception:
                logger.error("Monitor read failed")
            time.sleep(self.interval)


class AdbMonitorPower(_BaseAdbMonitor):
    """
    Monitor device power consumption via ADB.

    This monitor periodically reads power values from the hwmon interfaces
    on the device (`/sys/class/hwmon/hwmonN/power1_input`) using the provided
    `adb_handler`. Each sample is stored as a tuple `(power0, power1)` in watts.
    """

    def __init__(self, adb_handler, interval: float = 0.5) -> None:
        super().__init__(adb_handler=adb_handler, interval=interval)
        self.hwmon0_exists = self._check_hwmon("hwmon0")
        self.hwmon1_exists = self._check_hwmon("hwmon1")
        if self.hwmon0_exists and self.hwmon1_exists:
            self.idle_power_system: float = 0.0
            self.idle_power_processor: float = 0.0
            self.set_idle_power()

    def _check_hwmon(self, hwmon: str) -> bool:
        """
        Check if a hwmon device exposes a `power1_input` file.

        Parameters
        ----------
        hwmon : str
            Name of the hwmon node (e.g. "hwmon0", "hwmon1").

        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """
        try:
            self.adb_handler.shell(f"ls /sys/class/hwmon/{hwmon}/power1_input")
            time.sleep(self.interval)
            return True
        except Exception:
            logger.warning(
                f"Hardware monitoring device {hwmon} missing. "
                f"Proceeding without {hwmon} power monitoring."
            )
            return False

    def _read_hwmon(self, hwmon: str) -> float | None:
        """
        Read instantaneous power from a single hwmon node.

        Parameters
        ----------
        hwmon : str
            Name of the hwmon node (e.g. "hwmon0").

        Returns
        -------
        float | None
            Power reading in watts, or None if the read fails.
        """
        try:
            _, out, _ = self.adb_handler.shell(
                f"cat /sys/class/hwmon/{hwmon}/power1_input"
            )
            return int(out) / 1_000_000  # µW → W
        except Exception:
            logger.warning(f"Failed to read {hwmon} power value.")
            return None

    def _read(self) -> tuple[float | None, float | None] | None:
        """
        Read instantaneous power consumption from both hwmon devices.

        Returns
        -------
        tuple[float | None, float | None] | None
            A tuple `(power0, power1)` in watts, where each element may
            be None if the associated hwmon device is missing or the read
            fails. Returns None only if both devices are unavailable.
        """
        power0 = self._read_hwmon("hwmon0") if self.hwmon0_exists else None
        power1 = self._read_hwmon("hwmon1") if self.hwmon1_exists else None

        if power0 is None and power1 is None:
            return None
        return (power0, power1)

    def get_stats(self) -> tuple[float | None, float | None]:
        """
        Compute average power over all collected samples.

        Returns
        -------
        tuple[float | None, float | None]
            `(mean_power0, mean_power1)` in watts. Each value is None if no
            valid samples were collected for that channel.
        """
        samples = [m for m in self.measurements if isinstance(m, tuple) and len(m) == 2]
        p0_vals = [p0 for (p0, _) in samples if p0 is not None and p0 > self.idle_power_system]
        p1_vals = [p1 for (_, p1) in samples if p1 is not None and p1 > self.idle_power_processor]

        mean_p0 = statistics.fmean(p0_vals) if p0_vals else None
        mean_p1 = statistics.fmean(p1_vals) if p1_vals else None
        return mean_p0, mean_p1

    def set_idle_power(self) -> None:
        logger.info("Calculating idle power consumption...")
        self.start()
        time.sleep(5)
        self.stop()

        self.idle_power_system, self.idle_power_processor = self.get_stats() or (0.0, 0.0)
        self.idle_power_system *= 1.1  # add 10% margin
        self.idle_power_processor *= 1.1  # add 10% margin
        logger.info(
            f"Idle power consumption: system={self.idle_power_system:.4f} W, processor={self.idle_power_processor:.4f} W"
        )

class AdbMonitorDSP(_BaseAdbMonitor):
    """
    Monitor DSP utilization via ADB helper script on the device.

    This monitor periodically runs `/data/local/oak_dsp_util.sh` (or an
    equivalent utility) via `adb_handler` and stores the reported DSP
    utilization as a float in the range [0, 100].
    """

    def __init__(self, adb_handler, interval: float = 0.5) -> None:
        super().__init__(adb_handler=adb_handler, interval=interval)
        self.dsp_exists = self._check_dsp()
        if self.dsp_exists:
            self.idle_dsp_utilization: float = 0.0
            self.set_idle_dsp()

    def _prepare_dsp_util_script(self) -> None:
        """
        Create the DSP utility script directly on the device via ADB.

        """
        remote_script_path = "/data/local/oak_dsp_util.sh"

        script_content = r'''SYS_MON_APP="/usr/bin/sysMonApp"
SLEEP_TIME=1.0

$SYS_MON_APP getPowerStats --clear 1 --q6 cdsp >/dev/null 2>&1

$SYS_MON_APP getPowerStats --q6 cdsp >/data/local/dsp_read1_full 2>/dev/null

grep '^[[:space:]]*[0-9]*\.[0-9]*[[:space:]]*[0-9]*\.[0-9]*' \
    /data/local/dsp_read1_full > /data/local/dsp_read1

sleep $SLEEP_TIME

$SYS_MON_APP getPowerStats --q6 cdsp > /data/local/dsp_read2_full 2>/dev/null

grep '^[[:space:]]*[0-9]*\.[0-9]*[[:space:]]*[0-9]*\.[0-9]*' \
    /data/local/dsp_read2_full > /data/local/dsp_read2

dsp_util=$(
awk -v interval=$SLEEP_TIME '
    FILENAME == ARGV[1] && FNR > 1 {
        gsub(/^[[:blank:]]+/, "", $0);
        if (NF >= 2) {
            freq = $1;
            active1[freq] = $2;
            all_freqs[freq] = 1;
        }
    }
    FILENAME == ARGV[2] && FNR > 1 {
        gsub(/^[[:blank:]]+/, "", $0);
        if (NF >= 2) {
            freq = $1;
            active2[freq] = $2;
            all_freqs[freq] = 1;
        }
    }
    END {
        sum_delta = 0;
        max_freq = 0;
        delete deltas;
        for (freq in all_freqs) {
            f = freq + 0;
            if (f > max_freq) max_freq = f;
            a1 = (freq in active1) ? active1[freq] + 0 : 0;
            a2 = (freq in active2) ? active2[freq] + 0 : 0;
            delta = a2 - a1;
            if (delta < 0) {
                delta = 0;
            }
            deltas[freq] = delta;
            sum_delta += delta;
        }

        scale_factor = 1;
        if (sum_delta > interval) {
            scale_factor = interval / sum_delta;
        }

        sum_cycles = 0;
        for (freq in all_freqs) {
            f = freq + 0;
            adjusted_delta = deltas[freq] * scale_factor;
            sum_cycles += f * adjusted_delta;
        }

        if (max_freq == 0) {
            print "Error: Maximum frequency is zero." > "/dev/stderr";
            exit 1;
        }

        utilization = (sum_cycles / (max_freq * interval)) * 100;
        print utilization
    }
    ' /data/local/dsp_read1 /data/local/dsp_read2
)

echo "$dsp_util"
'''

        cmd = f"cat > {remote_script_path} <<'EOF'\n{script_content}\nEOF\n"
        self.adb_handler.shell(cmd)
        self.adb_handler.shell(f"chmod +x {remote_script_path}")

    def _check_dsp(self) -> bool:
        """
        Check if any supported DSP utility script exists on the device.

        Returns
        -------
        bool
            True if a supported utility is present, False otherwise.
        """
        try:
            self._prepare_dsp_util_script()
            self.adb_handler.shell(
                "ls -d /data/local/oak_dsp_util.sh /usr/bin/sysMonApp"
            )
            time.sleep(self.interval)
            return True
        except Exception:
            logger.warning(
                "No DSP utility scripts found. Proceeding without DSP monitoring."
            )
            return False

    def _read(self) -> float | None:
        """
        Read instantaneous DSP utilization from the device.

        Returns
        -------
        float | None
            DSP utilization as a percentage in the range [0, 100], or None
            if the value cannot be read or the DSP utility is unavailable.
        """
        if not self.dsp_exists:
            return None

        try:
            _, dsp, error = self.adb_handler.shell("bash /data/local/oak_dsp_util.sh")
            if "Maximum frequency is zero" in error:
                dsp = 0
            return float(dsp)
        except Exception:
            logger.warning("Failed to read DSP value.")
            return None

    def stop(self, full_cleanup: bool = False) -> None:
        """
        Stop the monitor and clean up temporary files on the device.
        """
        super().stop()
        if self.dsp_exists:
            files_to_remove = [
                "/data/local/dsp_read1",
                "/data/local/dsp_read2",
                "/data/local/dsp_read1_full",
                "/data/local/dsp_read2_full",
            ]
            if full_cleanup:
                files_to_remove.append("/data/local/oak_dsp_util.sh")

            returncode, _, _ = self.adb_handler.shell(f"rm -f {' '.join(files_to_remove)}")
            if returncode != 0:
                logger.warning("Failed to cleanup DSP monitor tmp files")

    def get_stats(self) -> float | None:
        """
        Compute average DSP utilization over all collected samples.

        Returns
        -------
        float | None
            Mean DSP utilization (0–100). None if no valid samples were
            collected.
        """
        dsp_vals = [dsp for dsp in self.measurements if isinstance(dsp, (int, float)) and dsp > self.idle_dsp_utilization]

        return statistics.fmean(dsp_vals) if dsp_vals else None

    def set_idle_dsp(self) -> None:
        logger.info("Calculating idle DSP utilization...")
        self.start()
        time.sleep(5)
        self.stop()

        self.idle_dsp_utilization = self.get_stats() or 0.0
        self.idle_dsp_utilization *= 1.1  # add 10% margin
        logger.info(f"Idle DSP utilization: {self.idle_dsp_utilization:.4f}%")

def mxid_to_adb_id(mxid: str) -> str:
    if mxid.isdigit():
        return format(int(mxid), "x")
    return mxid.encode("ascii").hex()

def get_device_info(device_ip: str | None, device_mxid: str | None) -> tuple[str | None, str | None]:
    if not device_ip and not device_mxid:
        return None, None

    if device_mxid:
        for info in dai.Device.getAllAvailableDevices():
            if device_mxid == info.getDeviceId():
                if device_ip and device_ip != info.name:
                    logger.warning(
                        f"Both device_mxid and device_ip provided, but they refer to different devices. Using device with device_mxid: {device_mxid} and device_ip: {info.name}."
                    )
                return info.name, mxid_to_adb_id(device_mxid)
    if device_ip:
        with dai.Device(device_ip) as device:
            inferred_mxid = device.getDeviceId()
            return device_ip, mxid_to_adb_id(inferred_mxid)

    return None, None