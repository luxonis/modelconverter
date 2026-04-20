import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from loguru import logger
from typing_extensions import override

from modelconverter.utils import (
    DeviceHandler,
)

DSP_UTIL_SCRIPT_CONTENT = r"""SYS_MON_APP="/usr/bin/sysMonApp"
SLEEP_TIME=1.0

$SYS_MON_APP getPowerStats --clear 1 --q6 cdsp >/dev/null 2>&1

$SYS_MON_APP getPowerStats --q6 cdsp >/data/modelconverter/dsp_read1_full 2>/dev/null

grep '^[[:space:]]*[0-9]*\.[0-9]*[[:space:]]*[0-9]*\.[0-9]*' \
    /data/modelconverter/dsp_read1_full > /data/modelconverter/dsp_read1

sleep $SLEEP_TIME

$SYS_MON_APP getPowerStats --q6 cdsp > /data/modelconverter/dsp_read2_full 2>/dev/null

grep '^[[:space:]]*[0-9]*\.[0-9]*[[:space:]]*[0-9]*\.[0-9]*' \
    /data/modelconverter/dsp_read2_full > /data/modelconverter/dsp_read2

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
    ' /data/modelconverter/dsp_read1 /data/modelconverter/dsp_read2
)

echo "$dsp_util"
"""


@dataclass
class BaseMonitor(ABC):
    """Base class for simple threaded device monitors.

    Subclasses implement C{_read()} to return a single sample value or
    C{None}. The base class handles starting and stopping a background
    sampling thread and storing collected measurements.

    @ivar device_handler: Handler used to communicate with the target
        device.
    @type device_handler: DeviceHandler
    @ivar interval: Sampling interval in seconds.
    @type interval: float
    """

    device_handler: DeviceHandler
    interval: float = 0.5

    _measurements: list[object] | None = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    @abstractmethod
    def read(self) -> object | None:
        """Perform a single sampling operation.

        @return: A single measurement value, with the concrete type
            defined by the subclass, or C{None} if the sample could not
            be read.
        @rtype: object | None
        """
        ...

    @property
    def measurements(self) -> Sequence[object]:
        """Sequence of all successfully collected measurements.

        @return: A list-like view of all values returned by C{_read()},
            excluding C{None}. Empty if the monitor has not run or if
            all reads failed.
        @rtype: Sequence[object]
        """
        return self._measurements or []

    def start(self) -> None:
        """Start the background sampling thread.

        If the monitor is already running, this method does nothing.
        """
        if self._running:
            return
        time.sleep(1)  # Small delay to avoid overlapping ADB commands
        self._measurements = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background sampling thread and wait for it to
        finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join()

    def _loop(self) -> None:
        """Internal sampling loop executed in the background thread."""
        assert self._measurements is not None
        while self._running:
            try:
                val = self.read()
                if val is not None:
                    self._measurements.append(val)
            except Exception:
                logger.error("Monitor read failed")
            time.sleep(self.interval)


class MonitorPower(BaseMonitor):
    """Monitor device power consumption via a device handler.

    This monitor periodically reads power values from the hwmon
    interfaces on the device, located at
    C{/sys/class/hwmon/hwmonN/power1_input}, using the configured
    C{device_handler}. Each sample is stored as a tuple C{(power0,
    power1)} in watts.
    """

    def __init__(
        self, device_handler: DeviceHandler, interval: float = 0.5
    ) -> None:
        """Initialize the power monitor.

        @param device_handler: Handler used to communicate with the
            target device.
        @type device_handler: DeviceHandler
        @param interval: Sampling interval in seconds.
        @type interval: float
        """
        super().__init__(device_handler=device_handler, interval=interval)
        self.hwmon0_exists = self._check_hwmon("hwmon0")
        self.hwmon1_exists = self._check_hwmon("hwmon1")
        if self.hwmon0_exists and self.hwmon1_exists:
            self.idle_power_system: float = 0.0
            self.idle_power_processor: float = 0.0
            self.set_idle_power()

    @override
    def read(self) -> tuple[float | None, float | None] | None:
        """Read instantaneous power consumption from both hwmon devices.

        @return: A tuple C{(power0, power1)} in watts, where each
            element may be C{None} if the associated hwmon device is
            missing or the read fails. Returns C{None} only if both
            devices are unavailable.
        @rtype: tuple[float | None, float | None] | None
        """
        power0 = self._read_hwmon("hwmon0") if self.hwmon0_exists else None
        power1 = self._read_hwmon("hwmon1") if self.hwmon1_exists else None

        if power0 is None and power1 is None:
            return None
        return (power0, power1)

    def get_stats(self) -> tuple[float | None, float | None]:
        """Compute average power over all collected samples.

        @return: A tuple C{(mean_power0, mean_power1)} in watts. Each
            value is C{None} if no valid samples were collected for that
            channel.
        @rtype: tuple[float | None, float | None]
        """
        samples = [
            m
            for m in self.measurements
            if isinstance(m, tuple) and len(m) == 2
        ]
        p0_vals = [
            p0
            for (p0, _) in samples
            if p0 is not None and p0 > self.idle_power_system
        ]
        p1_vals = [
            p1
            for (_, p1) in samples
            if p1 is not None and p1 > self.idle_power_processor
        ]

        mean_p0 = statistics.fmean(p0_vals) if p0_vals else None
        mean_p1 = statistics.fmean(p1_vals) if p1_vals else None
        return mean_p0, mean_p1

    def set_idle_power(self) -> None:
        """Measure and store baseline idle power consumption.

        The method samples power for a short period, computes average
        idle values for system and processor channels, and applies a 10
        percent margin to both baselines.
        """
        logger.info("Calculating idle power consumption...")
        self.start()
        time.sleep(5)
        self.stop()

        power_stats = self.get_stats()
        self.idle_power_system = power_stats[0] or 0.0
        self.idle_power_processor = power_stats[1] or 0.0
        self.idle_power_system *= 1.1  # add 10% margin
        self.idle_power_processor *= 1.1  # add 10% margin
        logger.info(
            f"Idle power consumption: system={self.idle_power_system:.4f} W, processor={self.idle_power_processor:.4f} W"
        )

    def _check_hwmon(self, hwmon: str) -> bool:
        """Check whether a hwmon device exposes a C{power1_input} file.

        @param hwmon: Name of the hwmon node, for example C{"hwmon0"} or
            C{"hwmon1"}.
        @type hwmon: str
        @return: C{True} if the file exists, otherwise C{False}.
        @rtype: bool
        """
        try:
            self.device_handler.shell(
                f"ls /sys/class/hwmon/{hwmon}/power1_input"
            )
        except Exception:
            logger.warning(
                f"Hardware monitoring device {hwmon} missing. "
                f"Proceeding without {hwmon} power monitoring."
            )
            return False
        return True

    def _read_hwmon(self, hwmon: str) -> float | None:
        """Read instantaneous power from a single hwmon node.

        @param hwmon: Name of the hwmon node, for example C{"hwmon0"}.
        @type hwmon: str
        @return: Power reading in watts, or C{None} if the read fails.
        @rtype: float | None
        """
        try:
            _, out, _ = self.device_handler.shell(
                f"cat /sys/class/hwmon/{hwmon}/power1_input"
            )
            return int(out) / 1_000_000  # µW -> W
        except Exception:
            logger.warning(f"Failed to read {hwmon} power value.")
            return None


class MonitorDSP(BaseMonitor):
    """Monitor DSP utilization via a helper script on the device.

    This monitor periodically runs
    C{/data/modelconverter/oak_dsp_util.sh}, or an equivalent utility,
    using C{device_handler} and stores the reported DSP utilization as a
    float in the range C{[0, 100]}.
    """

    def __init__(
        self, device_handler: DeviceHandler, interval: float = 0.5
    ) -> None:
        """Initialize the DSP monitor.

        @param device_handler: Handler used to communicate with the
            target device.
        @type device_handler: DeviceHandler
        @param interval: Sampling interval in seconds.
        @type interval: float
        """
        super().__init__(device_handler=device_handler, interval=interval)
        self.dsp_exists = self._check_dsp()
        if self.dsp_exists:
            self.idle_dsp_utilization: float = 0.0
            self.set_idle_dsp()

    @override
    def read(self) -> float | None:
        """Read instantaneous DSP utilization from the device.

        @return: DSP utilization as a percentage in the range C{[0,
            100]}, or C{None} if the value cannot be read or the DSP
            utility is unavailable.
        @rtype: float | None
        """
        if not self.dsp_exists:
            return None

        try:
            _, dsp, error = self.device_handler.shell(
                "bash /data/modelconverter/oak_dsp_util.sh"
            )
            if "Maximum frequency is zero" in error:
                dsp = 0
            return float(dsp)
        except Exception:
            logger.warning("Failed to read DSP value.")
            return None

    @override
    def stop(self, full_cleanup: bool = False) -> None:
        """Stop the monitor and optionally clean up temporary device
        files.

        @param full_cleanup: If C{True}, also remove the DSP utility
            script from the device. Otherwise only temporary readout
            files are removed.
        @type full_cleanup: bool
        """
        super().stop()
        if self.dsp_exists:
            files_to_remove = [
                "/data/modelconverter/dsp_read1",
                "/data/modelconverter/dsp_read2",
                "/data/modelconverter/dsp_read1_full",
                "/data/modelconverter/dsp_read2_full",
            ]
            if full_cleanup:
                files_to_remove.append("/data/modelconverter/oak_dsp_util.sh")

            returncode, _, _ = self.device_handler.shell(
                f"rm -f {' '.join(files_to_remove)}"
            )
            if returncode != 0:
                logger.warning("Failed to cleanup DSP monitor tmp files")

    def get_stats(self) -> float | None:
        """Compute average DSP utilization over all collected samples.

        @return: Mean DSP utilization in the range C{0-100}, or C{None}
            if no valid samples were collected.
        @rtype: float | None
        """
        dsp_vals = [
            dsp
            for dsp in self.measurements
            if isinstance(dsp, int | float) and dsp > self.idle_dsp_utilization
        ]

        return statistics.fmean(dsp_vals) if dsp_vals else None

    def set_idle_dsp(self) -> None:
        """Measure and store baseline idle DSP utilization.

        The method samples DSP utilization for a short period, computes
        the average idle utilization, and applies a 10 percent margin.
        """
        logger.info("Calculating idle DSP utilization...")
        self.start()
        time.sleep(5)
        self.stop()

        self.idle_dsp_utilization = self.get_stats() or 0.0
        self.idle_dsp_utilization *= 1.1  # add 10% margin
        logger.info(f"Idle DSP utilization: {self.idle_dsp_utilization:.4f}%")

    def _prepare_dsp_util_script(self) -> None:
        """Create the DSP utility script directly on the target
        device."""
        remote_script_path = "/data/modelconverter/oak_dsp_util.sh"
        self.device_handler.shell("mkdir -p /data/modelconverter")

        cmd = f"cat > {remote_script_path} <<'EOF'\n{DSP_UTIL_SCRIPT_CONTENT}\nEOF\n"
        self.device_handler.shell(cmd)
        self.device_handler.shell(f"chmod +x {remote_script_path}")

    def _check_dsp(self) -> bool:
        """Check whether any supported DSP utility script exists.

        @return: C{True} if a supported utility is present, otherwise
            C{False}.
        @rtype: bool
        """
        try:
            self._prepare_dsp_util_script()
            self.device_handler.shell(
                "ls -d /data/modelconverter/oak_dsp_util.sh /usr/bin/sysMonApp"
            )
        except Exception:
            logger.warning(
                "No DSP utility script found under /usr/bin/sysMonApp. Consider updating the device OS. Proceeding without DSP monitoring."
            )
            return False
        return True
