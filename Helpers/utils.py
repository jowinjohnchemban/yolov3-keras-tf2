def ratios_to_coordinates(bx, by, bw, bh, width, height):
    """
    Convert relative coordinates to actual coordinates.
    Args:
        bx: Relative center x coordinate.
        by: Relative center y coordinate.
        bw: Relative box width.
        bh: Relative box height.
        width: Image batch width.
        height: Image batch height.

    Return:
        x1: x coordinate.
        y1: y coordinate.
        x2: x1 + Bounding box width.
        y2: y1 + Bounding box height.
    """
    w, h = bw * width, bh * height
    x, y = bx * width + (w / 2), by * height + (h / 2)
    return x, y, x + w, y + h