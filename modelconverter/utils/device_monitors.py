"""Sampling of a device's hardware counters during a benchmark.

Benchmarking a converted model on real hardware says how fast it runs
but not what it costs. The monitor here fills that gap: while the
benchmark is running, it polls the power, RAM, CPU, DSP and thermal
counters the device exposes over its shell, and reports their aggregates
afterwards. It is used by the RVC4 benchmark, which also takes an idle
baseline to compare against.
"""

import re
import statistics
import threading
import time
import types
from contextlib import suppress
from typing import Final, Literal

from loguru import logger
from typing_extensions import Self

from modelconverter.utils import DeviceHandler


class DeviceMonitor:
    """Background sampler of a device's hardware counters.

    Reads the power, RAM, CPU, DSP and temperature counters over the
    device's shell at a fixed interval from a daemon thread, keeping
    every sample so that it can be aggregated afterwards. A hardware
    monitor the device does not expose is detected once, up front, and
    its power counter is skipped from then on; the other counters are
    attempted on every sample and simply yield nothing when they fail.

    Doubles as a context manager: entering the ``with`` block starts the
    sampling thread and leaving it stops the thread.
    """

    _DSP_SYS_MON_APP: Final[str] = "/usr/bin/sysMonApp"

    def __init__(
        self,
        device_handler: DeviceHandler,
        interval: float = 0.5,
        model: Literal["4d", "4s", "4lite"] = "4lite",
    ) -> None:
        """Initialize the monitor and probe the available counters.

        The probing happens right away: the hardware monitors and the
        DSP utility are looked for over the device shell. A hardware
        monitor that is missing is left out of the sampling, while the
        outcome of the DSP probe is only recorded.

        Args:
            device_handler: Handler used to run shell commands on the
                device.
            interval: Delay in seconds between two samples.
            model: Device model being monitored. Recorded only.

        """
        self._device_handler = device_handler
        self._interval = interval
        self._hwmon0_exists = self._check_hwmon("hwmon0")
        self._hwmon1_exists = self._check_hwmon("hwmon1")
        self._dsp_exists = self._check_dsp()
        self._model = model

        self._measurements: dict[str, list[float]] = {}
        self._running = False
        self._thread = None

        # Previous /proc/stat snapshot for CPU utilization calculation
        self._prev_cpu_times: tuple[int, int] | None = None

    def __enter__(self) -> Self:
        """Start the sampling thread on entering the ``with`` block.

        Returns:
            The monitor itself.

        """
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Stop the sampling thread on leaving the ``with`` block.

        Args:
            exc_type: Type of the exception that left the block, if any.
            exc_val: Exception that left the block, if any.
            exc_tb: Traceback of the exception that left the block, if
                any.

        """
        self.stop()

    def _read(self) -> dict[str, float | None]:
        """Take one sample of every counter.

        Returns:
            Mapping of counter name to its value, with ``None`` for a
            counter that could not be read.

        """
        return (
            self._read_power()
            | self._read_ram()
            | self._read_cpu()
            | self._read_dsp()
            | self._read_temps()
        )

    def get_stats(self) -> dict[str, float | None]:
        """Aggregate the samples collected so far.

        The DSP frequency residencies and the DSP power collapse counter
        are summed, every other counter is averaged.

        Returns:
            Mapping of counter name to its aggregate, with ``None`` for
            a counter that has no samples.

        """
        stats = {}

        for key, values in self._measurements.items():
            if "dsp_freq_" in key or "power_collapse" in key:
                stats[key] = sum(values)
            else:
                stats[key] = statistics.fmean(values) if values else None

        return stats

    def start(self) -> None:
        """Start the background sampling thread.

        If the monitor is already running, this is a no-op.
        """
        if self._running:
            return
        time.sleep(1)  # Small delay to avoid overlapping ADB commands
        self._reset()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _reset(self) -> None:
        """Drop the collected samples and the CPU utilization
        baseline.
        """
        self._measurements = {}
        self._prev_cpu_times = None

    def stop(self) -> None:
        """Stop the background sampling thread and wait for it to
        finish.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join()

    def _loop(self) -> None:
        """Run the internal sampling loop in the background thread."""
        while self._running:
            try:
                val = self._read()
                for key, value in val.items():
                    if value is not None:
                        self._measurements.setdefault(key, []).append(value)
            except Exception as e:
                logger.exception("Monitor read failed")
                logger.debug(f"Monitor read error details: {e}")
            time.sleep(self._interval)

    def _read_temps(self) -> dict[str, float | None]:
        """Read the temperature of the device's thermal zones.

        Returns:
            Mapping of ``temp_zone<n>``, for the zones 92 to 96, to the
            temperature in degrees Celsius, plus ``temp_avg`` holding
            the mean over the zones that could be read.

        """
        temps = {
            f"temp_zone{zone}": self._read_temp(zone) for zone in range(92, 97)
        }
        temps["temp_avg"] = (
            sum(values := [t for t in temps.values() if t is not None])
            / len(values)
            if any(t is not None for t in temps.values())
            else None
        )
        return temps

    def _read_temp(self, zone: int) -> float | None:
        try:
            _, out, _ = self._device_handler.shell(
                f"cat /sys/class/thermal/thermal_zone{zone}/temp"
            )
            temp = int(out) / 1000  # m°C -> °C
            if temp < 0:
                return None
        except Exception:
            logger.warning("Failed to read temperature value.")
            return None
        else:
            return temp

    def _read_hwmon(self, hwmon: str) -> float | None:
        if hwmon == "hwmon0" and not self._hwmon0_exists:
            return None
        if hwmon == "hwmon1" and not self._hwmon1_exists:
            return None
        try:
            _, out, _ = self._device_handler.shell(
                f"cat /sys/class/hwmon/{hwmon}/power1_input"
            )
            return int(out) / 1_000_000  # µW -> W
        except Exception:
            logger.warning(f"Failed to read {hwmon} power value.")
            return None

    def _read_cpu_frequency(self) -> float | None:
        """Read the current clock frequency of the first CPU core.

        Returns:
            The frequency in MHz, or ``None`` if it could not be read.

        """
        try:
            _, out, _ = self._device_handler.shell(
                "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
            )
            return int(out) / 1000  # kHz -> MHz
        except Exception as e:
            logger.warning("Failed to read processor frequency.")
            logger.debug(f"Processor frequency read error details: {e}")
            return None

    def _read_ram(self) -> dict[str, float | None]:
        """Return used RAM in MiB."""
        try:
            _, out, _ = self._device_handler.shell("cat /proc/meminfo")
            meminfo = {}

            for line in out.splitlines():
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                value_part = parts[1].strip().split()[0]
                meminfo[key] = int(value_part)  # kB

            mem_total = meminfo.get("MemTotal")
            mem_available = meminfo.get("MemAvailable")

            if mem_total is None or mem_available is None:
                logger.warning("Failed to parse RAM info from /proc/meminfo.")
                return {"ram_used": None}

            used_kib = mem_total - mem_available
            return {"ram_used": used_kib / 1024}
        except Exception as e:
            logger.warning("Failed to read RAM usage.")
            logger.debug(f"RAM read error details: {e}")
            return {"ram_used": None}

    def _read_cpu_utilization(self) -> float | None:
        """Return total CPU utilization in percent based on /proc/stat
        deltas.

        The first call returns None because a previous sample is needed.
        """
        try:
            _, out, _ = self._device_handler.shell("cat /proc/stat")
            first_line = out.splitlines()[0].strip()
            parts = first_line.split()

            if not parts or parts[0] != "cpu" or len(parts) < 5:
                logger.warning("Failed to parse CPU info from /proc/stat.")
                return None

            values = [int(v) for v in parts[1:]]
            total = sum(values)

            # Linux aggregate idle time = idle + iowait
            idle = values[3]
            if len(values) > 4:
                idle += values[4]

            current = (total, idle)

            if self._prev_cpu_times is None:
                self._prev_cpu_times = current
                return None

            prev_total, prev_idle = self._prev_cpu_times
            self._prev_cpu_times = current

            delta_total = total - prev_total
            delta_idle = idle - prev_idle

            if delta_total <= 0:
                return None

            utilization = 100.0 * (1.0 - (delta_idle / delta_total))
            return max(0.0, min(100.0, utilization))
        except Exception as e:
            logger.warning("Failed to read CPU usage.")
            logger.debug(f"CPU read error details: {e}")
            return None

    def _read_cpu(self) -> dict[str, float | None]:
        """Read the CPU frequency and utilization in one go.

        Returns:
            Mapping with the ``cpu_frequency`` and ``cpu_utilization``
            counters.

        """
        return {
            "cpu_frequency": self._read_cpu_frequency(),
            "cpu_utilization": self._read_cpu_utilization(),
        }

    def _read_dsp(self) -> dict[str, float | None]:
        """Read the DSP power statistics and clear them on the device.

        The frequency residency histogram reported by ``sysMonApp`` is
        collected and then reset, so each call only covers the time
        since the previous one.

        Returns:
            Mapping with ``dsp_utilization``, ``dsp_avg_frequency`` and
            ``dsp_power_collapse``, plus one ``dsp_freq_<freq>`` entry
            per histogram bucket. Empty if the statistics could not be
            read.

        """

        def parse_freq_file(
            text: str,
        ) -> tuple[dict[str, float], float | None, float | None]:
            """Extract lines of the form ``<float> <float>``.

            Args:
                text: Raw text to parse.

            Returns:
                A tuple of the ``{freq: value}`` mapping, the power
                collapse value and the total time.

            """
            data = {}
            pattern = re.compile(r"^\s*([0-9]*\.[0-9]+)\s+([0-9]*\.[0-9]+)")

            power_collapse = None
            total_time = None
            for line in text.splitlines()[1:]:  # skip header
                match = pattern.match(line)
                if match:
                    freq = match.group(1)
                    value = float(match.group(2))
                    data[freq] = value
                if "power collapse" in line.lower():
                    with suppress(Exception):
                        power_collapse = float(line.split()[-1])
                if "total time" in line.lower():
                    with suppress(Exception):
                        total_time = float(line.split()[-1])

            if power_collapse is None:
                logger.warning("Failed to parse DSP power collapse value.")
            return data, power_collapse, total_time

        try:
            _, out, _ = self._device_handler.shell(
                f"{self._DSP_SYS_MON_APP} getPowerStats --q6 cdsp"
            )
            self._device_handler.shell(
                f"{self._DSP_SYS_MON_APP} getPowerStats --clear 1 --q6 cdsp"
            )
            hist, power_collapse, total_time = parse_freq_file(out)
            total_time = total_time or sum(hist.values())
            avg_freq = (
                sum(float(freq) * value for freq, value in hist.items())
                / total_time
            )
            util = (avg_freq / max(float(freq) for freq in hist)) * 100

        except Exception as e:
            logger.warning("Failed to read DSP stats.")
            logger.debug(f"DSP read error details: {e}")
            return {}

        else:
            return {
                "dsp_utilization": util,
                "dsp_avg_frequency": avg_freq,
                "dsp_power_collapse": power_collapse,
            } | {
                f"dsp_freq_{freq.replace('.', '_')}": value
                for freq, value in hist.items()
            }

    def _check_hwmon(self, hwmon: str) -> bool:
        """Check whether a hardware monitor exposes a power reading.

        Args:
            hwmon: Name of the hardware monitor, e.g. ``"hwmon0"``.

        Returns:
            ``True`` if the monitor's ``power1_input`` file could be
            listed on the device, ``False`` otherwise.

        """
        try:
            self._device_handler.shell(
                f"ls /sys/class/hwmon/{hwmon}/power1_input"
            )
        except Exception as e:
            logger.warning(
                f"Hardware monitoring device {hwmon} missing. "
                f"Proceeding without {hwmon} power monitoring."
            )
            logger.debug(f"{hwmon} check error details: {e}")
            return False
        return True

    def _read_power(self) -> dict[str, float | None]:
        """Read the system and processor power draw.

        Returns:
            Mapping with the ``power_system`` and ``power_processor``
            counters in watts, with ``None`` where the corresponding
            hardware monitor is missing or unreadable.

        """
        system = self._read_hwmon("hwmon0")
        proc = self._read_hwmon("hwmon1")
        return {
            "power_system": system,
            "power_processor": proc,
        }

    def _check_dsp(self) -> bool:
        """Check whether the DSP monitoring utility is available.

        Returns:
            ``True`` if ``sysMonApp`` could be run on the device,
            ``False`` otherwise.

        """
        try:
            self._device_handler.shell(
                f"{self._DSP_SYS_MON_APP} getPowerStats --q6 cdsp --clear 1"
            )
        except Exception:
            logger.exception(
                "No DSP utility script found under /usr/bin/sysMonApp. Consider updating the device OS. Proceeding without DSP monitoring."
            )
            return False
        return True

    def get_idle_measurements(self, t: float = 5) -> dict[str, float | None]:
        """Sample the counters while the device is idle.

        Gives the baseline the measurements taken under load are
        compared against.

        Args:
            t: How long to sample for, in seconds.

        Returns:
            The aggregated counters, each key prefixed with ``idle_``.

        """
        with self:
            time.sleep(t)
        return {
            f"idle_{key}": value for key, value in self.get_stats().items()
        }
