from typing import List


def make_default_layout(shape: List[int]) -> str:
    """Creates a default layout for the given shape.

    Tries to guess most common layouts for the given shape pattern.
    Otherwise, uses the first free letter of the alphabet for each dimension.

    Example:
        >>> make_default_layout([1, 3, 256, 256])
        >>> "NCHW"
        >>> make_default_layout([1, 19, 7, 8])
        >>> "NABC"
    """
    layout = []
    i = 0
    if shape[0] == 1:
        layout.append("N")
        i += 1
    if len(shape) - i == 3:
        if shape[i] < shape[i + 1] and shape[i] < shape[i + 2]:
            return "".join(layout + ["C", "H", "W"])
        elif shape[-1] < shape[-2] and shape[-1] < shape[-3]:
            return "".join(layout + ["H", "W", "C"])
    i = 0
    while len(layout) < len(shape):
        # Starting with "C" for more sensible defaults
        letter = chr(ord("A") + (i + 2) % 26)
        if letter not in layout:
            layout.append(letter)
        i += 1
    return "".join(layout)


def guess_new_layout(
    old_layout: str, old_shape: List[int], new_shape: List[int]
) -> str:
    """Tries to guess the layout of the new shape.

    The new shape must contain the same elements as the old one.
    If two values are the same, the order of their labels will be preserved.

    Example:
        >>> old_shape = [1, 3, 256, 256]
        >>> old_layout = "NCHW"
        >>> guess_new_layout(old_layout, old_shape, [1, 256, 256, 3])
        >>> "NHWC"

    @type old_layout: str
    @param old_layout: Old layout

    @type old_shape: List[int]
    @param old_shape: Old shape

    @type other: List[int]
    @param other: New shape to guess the layout of

    @rtype: str
    @return: Lettercode representation of the new layout
    """
    if len(new_shape) != len(old_layout):
        raise ValueError(
            "The length of the new shape must be the same as the old one."
        )
    if sorted(old_shape) != sorted(new_shape):
        raise ValueError(
            "The new shape must contain the same elements as the old one."
        )
    old_shape_tuples = list(zip(old_layout, old_shape))

    new_layout = []
    for dim in new_shape:
        for i, (old_label, old_dim) in enumerate(old_shape_tuples):
            if old_dim == dim:
                new_layout.append(old_label)
                old_shape_tuples.pop(i)
                break

    return "".join(new_layout)
