import depthai as dai
from loguru import logger


def device_id_to_adb_id(device_id: str) -> str:
    if device_id.isdigit():
        return format(int(device_id), "x")
    return device_id.encode("ascii").hex()


def adb_id_to_device_id(adb_id: str) -> str:
    try:
        int_id = int(adb_id, 16)
        return str(int_id)
    except ValueError:
        bytes_id = bytes.fromhex(adb_id)
        return bytes_id.decode("ascii")


def get_device_info(
    device_ip: str | None, device_id: str | None
) -> tuple[str | None, str | None]:
    if not device_ip and not device_id:
        return None, None

    if device_id:
        if device_id.isdecimal():
            adb_id = device_id_to_adb_id(device_id)
        else:
            adb_id = device_id
            device_id = adb_id_to_device_id(adb_id)
        for info in dai.Device.getAllAvailableDevices():
            if device_id == info.getDeviceId():
                if device_ip and device_ip != info.name:
                    logger.warning(
                        f"Both device_id and device_ip provided, but they refer to different devices. Using device with device_id: {device_id} and device_ip: {info.name}."
                    )
                return info.name, adb_id
    if device_ip:
        with dai.Device(device_ip) as device:
            inferred_device_id = device.getDeviceId()
            return device_ip, device_id_to_adb_id(inferred_device_id)
    return None, None
