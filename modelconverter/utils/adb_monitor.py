import threading
import time
import statistics
from collections.abc import Sequence
from loguru import logger

class AdbMonitor:
    """
    Periodically collect metrics from a device over ADB.

    If enabled and available, this monitor runs a background thread that, at a fixed interval, reads:
      - Power values from hwmon0 and hwmon1.
      - DSP utilization via a utility script.

    Parameters
    ----------
    adb_handler : AdbHandler
        Android Debug Bridge (adb) CLI wrapper.
    period : float, optional
        Sampling period in seconds. Default is 0.1.
    monitor_power : bool, optional
        Whether to attempt reading power from hwmon. Default is False.
    monitor_dsp : bool, optional
        Whether to attempt reading DSP utilization. Default is False.
    """

    def __init__(
        self,
        adb_handler,
        period: float = 0.1,
        monitor_power: bool = False,
        monitor_dsp: bool = False,
    ) -> None:
        self.adb_handler = adb_handler
        self.period = period
        self.monitor_power = monitor_power
        self.monitor_dsp = monitor_dsp

        if self.monitor_power:
            self.hwmon0_exists, self.hwmon1_exists = self._check_hwmon()
        if self.monitor_dsp:
            self.dsp_exists = self._check_dsp()

        self._measurements: list[tuple[float | None, float | None]] | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def _check_hwmon(self) -> tuple[bool, bool]:
        """
        Ensure hwmon0 and hwmon1 have a power1_input file.
        """
        try:
            self.adb_handler.shell("ls /sys/class/hwmon/hwmon0/power1_input")
            exists0 = True
        except Exception:
            logger.warning(
                "Hardware monitoring (hwmon0) device missing. Proceeding without hwmon0 power monitoring."
            )
            exists0 = False

        try:
            self.adb_handler.shell("ls /sys/class/hwmon/hwmon1/power1_input")
            exists1 = True
        except Exception:
            logger.warning(
                "Hardware monitoring (hwmon1) device missing. Proceeding without hwmon1 power monitoring."
            )
            exists1 = False

        return exists0, exists1

    def _check_dsp(self) -> bool:
        """
        Ensure the required DSP utility script exists on the device.
        """
        try:
            self.adb_handler.shell(
                "ls -d /data/local/oak_dsp_util.sh /data/local/sysMonAppLE"
            )
            return True
        except:
            logger.warning(
                "No DSP utility scripts found. Proceeding without DSP monitoring."
            )
            return False

    def start(self) -> None:
        if self._running: 
            return # already running
        self._measurements = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()
        if self.monitor_dsp:
            returncode, _, _ = self.adb_handler.shell(
            "rm /data/local/dsp_read1 /data/local/dsp_read2 /data/local/dsp_read1_full /data/local/dsp_read2_full"
            )
            if not returncode == 0:
                logger.warning("Failed to cleanup DSP monitor tmp files")

    def _loop(self) -> None:
        assert self._measurements is not None
        while self._running:
            try:
                val = self._read()
                if val is not None:
                    self._measurements.append(val)
            except Exception:
                logger.error("Monitor read failed")
            time.sleep(self.period)

    def _read_power(self) -> tuple[float | None, float | None] | None:
        """
        Read instantaneous power consumption from the device via ADB.

        Values are read and converted from µW to W.
        Returns a tuple (power0, power1).
        """
        power0 = power1 = None

        if self.hwmon0_exists:
            try:
                _, out, _ = self.adb_handler.shell(
                    "cat /sys/class/hwmon/hwmon0/power1_input"
                )
                power0 = int(out) / 1_000_000
            except:
                logger.warning("Failed to read hwmon0 power value.")

        if self.hwmon1_exists:
            try:
                _, out, _ = self.adb_handler.shell(
                    "cat /sys/class/hwmon/hwmon1/power1_input"
                )
                power1 = int(out) / 1_000_000
            except:
                logger.warning("Failed to read hwmon1 power value.")

        return (power0, power1)

    def _read_dsp(self) -> float | None:
        """
        Read instantaneous DSP utilization from the device via ADB.

        Returns a float value between 0 and 100.
        """
        if self.dsp_exists:
            try:
                _, dsp, error = self.adb_handler.shell(
                    "bash /data/local/oak_dsp_util.sh"
                )
                if "Maximum frequency is zero" in error:
                    dsp = 0
                return float(dsp)
            except:
                logger.warning("Failed to read DSP value.")
        return None

    def _read(self) -> tuple[tuple[float | None, float | None], float | None] | None:
        """
        Read power and DSP values from the device.

        Returns a tuple ((power0, power1), dsp).
        """
        power0 = power1 = dsp = None
        if self.monitor_power:
            power0, power1 = self._read_power()
        if self.monitor_dsp:
            dsp = self._read_dsp()
        return (power0, power1), dsp
    
    @property
    def measurements(self) -> Sequence[tuple[float | None, float | None]]:
        return self._measurements or []

    def get_stats(self) -> dict[str, float]:
        """
        Return simple stats over the collected power and DSP samples.
        """
        p0_vals = [p0 for (p0, _), _ in self._measurements if p0 is not None]
        p1_vals = [p1 for (_, p1), _ in self._measurements if p1 is not None]
        dsp_vals = [dsp for _, dsp in self._measurements if dsp is not None]

        return {
            "power0": statistics.fmean(p0_vals) if p0_vals else -1,
            "power1": statistics.fmean(p1_vals) if p1_vals else -1,
            "dsp": statistics.fmean(dsp_vals) if dsp_vals else -1,
        }