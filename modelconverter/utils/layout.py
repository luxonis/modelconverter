"""Guessing of tensor layouts from tensor shapes.

A layout is a lettercode such as ``"NCHW"`` naming every dimension of
a tensor. Conversion needs one for each input and output -- to know
which axes are spatial, and how an image has to be laid out before it
is fed to the model -- but a model rarely states it, and the vendor
toolchains reorder the dimensions along the way. The helpers here
recover a plausible layout from a shape alone, and carry a known
layout over to a reordered shape.
"""


def make_default_layout(shape: list[int]) -> str:
    """Create a default layout for the given shape.

    Tries to guess most common layouts for the given shape pattern.
    Otherwise, uses the first free letter of the alphabet for each
    dimension.

    Args:
        shape: Shape to create the layout for.

    Returns:
        Lettercode representation of the layout.

    Example:
        >>> make_default_layout([1, 3, 256, 256])
        'NCHW'
        >>> make_default_layout([1, 19, 7, 8])
        'NCDE'

    """
    layout = []
    i = 0
    if shape[0] == 1:
        layout.append("N")
        i += 1
    if len(shape) - i == 3:
        if shape[i] < shape[i + 1] and shape[i] < shape[i + 2]:
            return "".join([*layout, "C", "H", "W"])
        if shape[-1] < shape[-2] and shape[-1] < shape[-3]:
            return "".join([*layout, "H", "W", "C"])
    i = 0
    while len(layout) < len(shape):
        # Starting with "C" for more sensible defaults
        letter = chr(ord("A") + (i + 2) % 26)
        if letter not in layout:
            layout.append(letter)
        i += 1
    return "".join(layout)


def guess_new_layout(
    old_layout: str, old_shape: list[int], new_shape: list[int]
) -> str | None:
    """Guess the layout of the new shape.

    The new shape must contain the same elements as the old one. If two
    values are the same, the order of their labels will be preserved.

    Args:
        old_layout: Layout that describes ``old_shape``.
        old_shape: Shape the layout is known for.
        new_shape: Reordering of ``old_shape`` to label.

    Returns:
        Lettercode representation of the new layout.

    Raises:
        ValueError: If the new shape has a different length than the old
            layout, or does not contain the same elements as the old
            shape.

    Example:
        >>> guess_new_layout("NCHW", [1, 3, 256, 256], [1, 256, 256, 3])
        'NHWC'

    """
    if len(new_shape) != len(old_layout):
        raise ValueError(
            "The length of the new shape must be the same as the old one"
        )
    if sorted(old_shape) != sorted(new_shape):
        raise ValueError(
            "The new shape must contain the same elements as the old one"
        )
    old_shape_tuples = list(zip(old_layout, old_shape, strict=True))

    new_layout = []
    for dim in new_shape:
        for i, (old_label, old_dim) in enumerate(old_shape_tuples):
            if old_dim == dim:
                new_layout.append(old_label)
                old_shape_tuples.pop(i)
                break

    return "".join(new_layout)
