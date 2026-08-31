"""Helpers for addressing an RVC4 device.

DepthAI and ADB name the same device differently, so benchmarking and
analysis have to translate between a DepthAI device ID and an ADB
serial, and work out an address for a device the user named by either.
"""

import depthai as dai
from loguru import logger


def device_id_to_adb_id(device_id: str) -> str:
    """Convert a DepthAI device ID into an ADB serial.

    Args:
        device_id: The device ID as DepthAI reports it.

    Returns:
        The hexadecimal form of the ID if it is a number, otherwise the
        hex encoding of its ASCII bytes.

    Example:
        >>> device_id_to_adb_id("1844301")
        '1c244d'

    """
    if device_id.isdigit():
        return format(int(device_id), "x")
    return device_id.encode("ascii").hex()


def adb_id_to_device_id(adb_id: str) -> str:
    """Convert an ADB serial back into a DepthAI device ID.

    Args:
        adb_id: The serial ADB knows the device by.

    Returns:
        The decimal form of the serial if it reads as a hexadecimal
        number, otherwise the ASCII text its bytes spell out.

    Example:
        >>> adb_id_to_device_id("1c244d")
        '1844301'

    """
    try:
        int_id = int(adb_id, 16)
        return str(int_id)
    except ValueError:
        bytes_id = bytes.fromhex(adb_id)
        return bytes_id.decode("ascii")


def get_device_info(
    device_ip: str | None, device_id: str | None
) -> tuple[str | None, str | None]:
    """Work out how to reach the device the user asked for.

    A device ID is looked up among the connected devices, and a mismatch
    between the address found there and the one that was passed is only
    warned about. An address that does not come out of that lookup is
    connected to directly, to read the ID off the device.

    Args:
        device_ip: Address of the device, if one was given.
        device_id: Device ID or ADB serial of the device, if one was
            given.

    Returns:
        The address and the ADB serial of the device. Both are ``None``
        when neither argument was given, and when only a device ID was
        given and no connected device matched it.

    """
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
