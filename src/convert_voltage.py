def convert_force_voltage(fin, force_scale, force_offset):
    voltage = (fin - force_offset) / (force_scale)
    return voltage