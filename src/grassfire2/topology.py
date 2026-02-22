def ccw(i: int) -> int:
    """Get index (0, 1 or 2) increased with one (counter-clockwise)"""
    return (i + 1) % 3


def cw(i: int) -> int:
    """Get index (0, 1 or 2) decreased with one (clockwise)"""
    return (i - 1) % 3


def apex(side: int) -> int:
    """Given a side, give the apex of the triangle"""
    return side % 3


def orig(side: int) -> int:
    """Given a side, give the origin of the triangle"""
    return (side + 1) % 3  # ccw(side)


def dest(side: int) -> int:
    """Given a side, give the destination of the triangle"""
    return (side - 1) % 3  # cw(side)
