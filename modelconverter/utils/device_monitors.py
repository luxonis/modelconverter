import statistics
import threading
import time
import types

from loguru import logger
from typing_extensions import Self

from modelconverter.utils import DeviceHandler

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


class DeviceMonitor:
    def __init__(
        self, device_handler: DeviceHandler, interval: float = 0.5
    ) -> None:
        self.device_handler = device_handler
        self.interval = interval
        self.hwmon0_exists = self.check_hwmon("hwmon0")
        self.hwmon1_exists = self.check_hwmon("hwmon1")
        self.dsp_exists = self.check_dsp()
        self.idle_dsp_utilization: float = 0.0
        self.idle_power_system: float = 0.0
        self.idle_power_processor: float = 0.0
        self.idle_ram_used: float = 0.0

        self._measurements: list[
            tuple[float | None, float | None, float | None, float | None]
        ] = []
        self._running = False
        self._thread = None

        self.set_idle_measurements()

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

    def read(
        self,
    ) -> tuple[float | None, float | None, float | None, float | None] | None:
        system, proc = self.read_power()
        dsp = self.read_dsp()
        ram = self.read_ram()
        return system, proc, dsp, ram

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

    def get_stats(self) -> dict:
        system = []
        proc = []
        dsp = []
        ram = []

        for s, p, d, r in self._measurements:
            if isinstance(s, (int, float)) and s > self.idle_power_system:
                system.append(s)
            if isinstance(p, (int, float)) and p > self.idle_power_processor:
                proc.append(p)
            if isinstance(d, (int, float)) and d > self.idle_dsp_utilization:
                dsp.append(d)
            if isinstance(r, (int, float)) and r > self.idle_ram_used:
                ram.append(r)

        system_stats = self._calc_stats(system)
        proc_stats = self._calc_stats(proc)
        dsp_stats = self._calc_stats(dsp)
        ram_stats = self._calc_stats(ram)

        return {
            "power_system": system_stats["mean"],
            "power_system_median": system_stats["median"],
            "power_system_peak": system_stats["peak"],
            "power_processor": proc_stats["mean"],
            "power_processor_median": proc_stats["median"],
            "power_processor_peak": proc_stats["peak"],
            "dsp": dsp_stats["mean"],
            "dsp_median": dsp_stats["median"],
            "dsp_peak": dsp_stats["peak"],
            "ram_used": ram_stats["mean"],
            "ram_used_median": ram_stats["median"],
            "ram_used_peak": ram_stats["peak"],
        }

    def start(self) -> None:
        """Start the background sampling thread.

        If the monitor is already running, this is a no-op.
        """
        if self._running:
            return
        time.sleep(1)  # Small delay to avoid overlapping ADB commands
        self._measurements = []
        self._running = True
        self._thread = threading.Thread(target=self.loop, daemon=True)
        self._thread.start()

    def stop(self, full_cleanup: bool = False) -> None:
        """Stop the background sampling thread and wait for it to
        finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join()
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
                f"rm -f {' '.join(files_to_remove)}", check=False
            )
            if returncode != 0:
                logger.warning("Failed to cleanup DSP monitor tmp files")

    def loop(self) -> None:
        """Internal sampling loop executed in the background thread."""
        while self._running:
            try:
                val = self.read()
                if val is not None:
                    self._measurements.append(val)
            except Exception:
                logger.error("Monitor read failed")
            time.sleep(self.interval)

    def read_hwmon(self, hwmon: str) -> float | None:
        try:
            _, out, _ = self.device_handler.shell(
                f"cat /sys/class/hwmon/{hwmon}/power1_input"
            )
            return int(out) / 1_000_000  # µW -> W
        except Exception:
            logger.warning(f"Failed to read {hwmon} power value.")
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
        power0 = self.read_hwmon("hwmon0") if self.hwmon0_exists else None
        power1 = self.read_hwmon("hwmon1") if self.hwmon1_exists else None
        return (power0, power1)

    def read_dsp(self) -> float | None:
        code, dsp, error = self.device_handler.shell(
            "bash /data/modelconverter/oak_dsp_util.sh", check=False
        )
        if "Maximum frequency is zero" in error:
            dsp = 0
        elif code != 0:
            logger.warning("Failed to read DSP value.")
            return None
        return float(dsp)

    def check_dsp(self) -> bool:
        try:
            self.prepare_dsp_util_script()
            self.device_handler.shell(
                "ls -d /data/modelconverter/oak_dsp_util.sh /usr/bin/sysMonApp"
            )
        except Exception:
            logger.warning(
                "No DSP utility script found under /usr/bin/sysMonApp. Consider updating the device OS. Proceeding without DSP monitoring."
            )
            return False
        return True

    def prepare_dsp_util_script(self) -> None:
        """Create the DSP utility script directly on the device via
        ADB."""
        remote_script_path = "/data/modelconverter/oak_dsp_util.sh"
        self.device_handler.shell("mkdir -p /data/modelconverter")

        cmd = f"cat > {remote_script_path} <<'EOF'\n{DSP_UTIL_SCRIPT_CONTENT}\nEOF\n"
        self.device_handler.shell(cmd)
        self.device_handler.shell(f"chmod +x {remote_script_path}")

    def set_idle_measurements(self) -> None:
        logger.info("Calculating idle power consumption...")
        self.start()
        time.sleep(5)
        self.stop()

        stats = self.get_stats()
        self.idle_power_system = stats["power_system"] or 0.0
        self.idle_power_processor = stats["power_processor"] or 0.0
        self.idle_ram_used = stats["ram_used"] or 0.0
        self.idle_dsp_utilization = self.get_stats()["dsp"] or 0.0
        self.idle_power_system *= 1.1
        self.idle_power_processor *= 1.1
        self.idle_dsp_utilization *= 1.1
        logger.info(
            f"Idle power consumption: system={self.idle_power_system:.4f} W, processor={self.idle_power_processor:.4f} W"
        )

        logger.info(f"Idle DSP utilization: {self.idle_dsp_utilization:.4f}%")
        logger.info(f"Idle RAM used: {self.idle_ram_used:.2f} MiB")
