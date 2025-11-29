import depthai as dai

def get_adb_id(device_ip: str) -> str:
    if device_ip is None:
        return None
    try:
        with dai.Device(device_ip) as device:
            device_mxid = int(device.getDeviceId())
    except:
        raise RuntimeError(f"Failed to connect to device: {device_ip}")
    return format(device_mxid, "x")  # abd_id is hex version of mxid