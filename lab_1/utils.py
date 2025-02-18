def dec2bin(bin, a=0, b=3):
    decimal_value = int(bin, 2)
    max_value = 2 ** len(bin) - 1
    return a + (decimal_value / max_value) * (b - a)