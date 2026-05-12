import re
import statistics
import threading
import time
import types
from typing import Literal, NamedTuple

from loguru import logger
from typing_extensions import Self

from modelconverter.utils import DeviceHandler


class Measurement(NamedTuple):
    power_system: float | None = None
    power_processor: float | None = None
    processor_frequency: float | None = None
    dsp_utilization: float | None = None
    dsp_frequency: float | None = None
    ram_used: float | None = None
    cpu_utilization: float | None = None
    temp_zone92: float | None = None
    temp_zone93: float | None = None
    temp_zone94: float | None = None
    temp_zone95: float | None = None
    temp_zone96: float | None = None
    temp_avg: float | None = None

    @classmethod
    def zero(cls) -> Self:
        return cls(
            power_system=0.0,
            power_processor=0.0,
            processor_frequency=0.0,
            dsp_utilization=0.0,
            dsp_frequency=0.0,
            ram_used=0.0,
            cpu_utilization=0.0,
            temp_zone92=0.0,
            temp_zone93=0.0,
            temp_zone94=0.0,
            temp_zone95=0.0,
            temp_zone96=0.0,
        )


class DeviceMonitor:
    def __init__(
        self,
        device_handler: DeviceHandler,
        interval: float = 0.5,
        model: Literal["4d", "4s", "4lite"] = "4lite",
    ) -> None:
        self.device_handler = device_handler
        self.interval = interval
        self.hwmon0_exists = self.check_hwmon("hwmon0")
        self.hwmon1_exists = self.check_hwmon("hwmon1")
        self.dsp_exists = self.check_dsp()
        self.model = model

        self.idle_measurements = Measurement.zero()
        self._measurements: list[Measurement] = []
        self._running = False
        self._thread = None

        # Previous /proc/stat snapshot for CPU utilization calculation
        self._prev_cpu_times: tuple[int, int] | None = None

    @property
    def measurements(self) -> dict[str, list[float]]:
        return {
            key: [
                value
                for m in self._measurements
                if (value := getattr(m, key, None)) is not None
            ]
            for key in Measurement._fields
        }

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.stop()

    def read(self) -> Measurement:
        return Measurement(
            *self.read_power(),
            self.read_processor_frequency(),
            *self.read_dsp(),
            self.read_ram(),
            self.read_cpu(),
            **self.read_temp(),
        )

    def get_stats(self) -> dict[str, float | None]:
        from collections import defaultdict

        values = defaultdict(list)

        for measurement in self._measurements:
            for field, value in measurement._asdict().items():
                if isinstance(value, (int, float)):
                    values[field].append(value)

        result = {}

        for field, vals in values.items():
            stats = self._calc_stats(vals)

            result[field] = stats["mean"]
            result[f"{field}_median"] = stats["median"]
            result[f"{field}_peak"] = stats["peak"]

        return result

    def start(self, set_idle: bool = True) -> None:
        """Start the background sampling thread.

        If the monitor is already running, this is a no-op.
        """
        if self._running:
            return
        time.sleep(1)  # Small delay to avoid overlapping ADB commands
        if set_idle:
            self.set_idle_measurements()
        self.reset()
        self._running = True
        self._thread = threading.Thread(target=self.loop, daemon=True)
        self._thread.start()

    def reset(self) -> None:
        self._measurements = []
        self._prev_cpu_times = None

    def stop(self) -> None:
        """Stop the background sampling thread and wait for it to
        finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join()

    def loop(self) -> None:
        """Internal sampling loop executed in the background thread."""
        while self._running:
            try:
                val = self.read()
                if val is not None:
                    self._measurements.append(val)
            except Exception:
                logger.exception("Monitor read failed")
            time.sleep(self.interval)

    def read_temp(self) -> dict[str, float | None]:
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
            _, out, _ = self.device_handler.shell(
                f"cat /sys/class/thermal/thermal_zone{zone}/temp"
            )
            return int(out) / 1000  # m°C -> °C
        except Exception:
            logger.warning("Failed to read temperature value.")
            return None

    def read_hwmon(self, hwmon: str) -> float | None:
        if hwmon == "hwmon0" and not self.hwmon0_exists:
            return None
        if hwmon == "hwmon1" and not self.hwmon1_exists:
            return None
        try:
            _, out, _ = self.device_handler.shell(
                f"cat /sys/class/hwmon/{hwmon}/power1_input"
            )
            return int(out) / 1_000_000  # µW -> W
        except Exception:
            logger.warning(f"Failed to read {hwmon} power value.")
            return None

    def read_processor_frequency(self) -> float | None:
        try:
            _, out, _ = self.device_handler.shell(
                "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
            )
            return int(out) / 1000  # kHz -> MHz
        except Exception:
            logger.warning("Failed to read processor frequency.")
            return None

    def read_ram(self) -> float | None:
        """Return used RAM in MiB."""
        try:
            _, out, _ = self.device_handler.shell("cat /proc/meminfo")
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
                return None

            used_kib = mem_total - mem_available
            return used_kib / 1024  # KiB -> MiB
        except Exception:
            logger.warning("Failed to read RAM usage.")
            return None

    def read_cpu(self) -> float | None:
        """Return total CPU utilization in percent based on /proc/stat
        deltas.

        The first call returns None because a previous sample is needed.
        """
        try:
            _, out, _ = self.device_handler.shell("cat /proc/stat")
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
        except Exception:
            logger.warning("Failed to read CPU usage.")
            return None

    def read_dsp(
        self,
    ) -> tuple[float | None, float | None]:

        SYS_MON_APP = "/usr/bin/sysMonApp"
        SLEEP_TIME = 1.0

        def parse_freq_file(text: str) -> dict[float, float]:
            """
            Extract lines like: <float> <float>
            Returns: {freq: value}
            """
            data = {}
            pattern = re.compile(r"^\s*([0-9]*\.[0-9]+)\s+([0-9]*\.[0-9]+)")

            for line in text.splitlines()[1:]:  # skip header
                match = pattern.match(line)
                if match:
                    freq = float(match.group(1))
                    value = float(match.group(2))
                    data[freq] = value

            return data

        def compute_utilization_and_freq(
            active1: dict[float, float],
            active2: dict[float, float],
            interval: float,
        ) -> tuple[float, float]:
            all_freqs = set(active1) | set(active2)

            deltas = {}
            sum_delta = 0
            max_freq = 0

            for freq in all_freqs:
                max_freq = max(max_freq, freq)

                a1 = active1.get(freq, 0)
                a2 = active2.get(freq, 0)

                delta = max(0, a2 - a1)
                deltas[freq] = delta
                sum_delta += delta

            # Normalize (same as AWK)
            scale_factor = 1.0
            if sum_delta > interval:
                scale_factor = interval / sum_delta

            sum_cycles = 0
            adjusted_total_time = 0

            for freq, delta in deltas.items():
                adjusted = delta * scale_factor
                sum_cycles += freq * adjusted
                adjusted_total_time += adjusted

            if max_freq == 0 or interval == 0:
                raise ValueError("Invalid data")

            utilization = (sum_cycles / (max_freq * interval)) * 100

            avg_freq = (
                sum_cycles / adjusted_total_time
                if adjusted_total_time > 0
                else 0
            )
            return utilization, avg_freq

        self.device_handler.shell(
            f"{SYS_MON_APP} getPowerStats --clear 1 --q6 cdsp"
        )

        _, out1, _ = self.device_handler.shell(
            f"{SYS_MON_APP} getPowerStats --q6 cdsp"
        )
        data1 = parse_freq_file(out1)

        time.sleep(SLEEP_TIME)

        _, out2, _ = self.device_handler.shell(
            f"{SYS_MON_APP} getPowerStats --q6 cdsp"
        )
        data2 = parse_freq_file(out2)

        try:
            return compute_utilization_and_freq(data1, data2, SLEEP_TIME)
        except Exception as e:
            logger.warning(f"Failed to compute DSP utilization: {e}")
            return None, None

    def check_hwmon(self, hwmon: str) -> bool:
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

    def read_power(self) -> tuple[float | None, float | None]:
        system = self.read_hwmon("hwmon0")
        proc = self.read_hwmon("hwmon1")
        return system, proc

    def check_dsp(self) -> bool:
        try:
            self.device_handler.shell("ls -d /usr/bin/sysMonApp")
        except Exception:
            logger.exception(
                "No DSP utility script found under /usr/bin/sysMonApp. Consider updating the device OS. Proceeding without DSP monitoring."
            )
            return False
        return True

    def set_idle_measurements(self) -> None:
        logger.info("Calculating idle power consumption...")
        self.start(set_idle=False)
        time.sleep(10)
        self.stop()

        stats = self.get_stats()
        self.idle_measurements = Measurement(
            *[stats.get(field) or 0.0 for field in Measurement._fields]
        )

        for field, value in self.idle_measurements._asdict().items():
            logger.info(f"Idle {field.replace('_', ' ')}: {value:.4f}")

    def _calc_stats(self, values: list[float]) -> dict[str, float | None]:
        if not values:
            return {
                "mean": None,
                "median": None,
                "peak": None,
            }

        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "peak": max(values),
        }
