from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass(init=False)
class Shape:
    """Class for handling shapes with labels.

    @type shape: List[int]
    @ivar shape: List of dimensions of the shape.

    @type layout: List[str]
    @ivar layout: List of labels for each dimension of the shape.
    """

    _shape: Dict[str, int]

    def __init__(
        self, shape: List[int], layout: Optional[Union[List[str], str]] = None
    ):
        if layout is None:
            layout = self._default_layout(shape)
        if isinstance(layout, str):
            layout = list(layout)

        if len(shape) != len(layout):
            raise ValueError("Length of shape and layout must be the same.")

        self._shape = OrderedDict(zip(layout, shape))
        self.layout = layout

    def __str__(self) -> str:
        return f"Shape({self.shape}, {self.layout})"

    def __repr__(self) -> str:
        return str(self)

    def __rich_repr__(self):
        yield "shape", self.shape
        yield "layout", self.layout_string

    def __len__(self) -> int:
        return len(self._shape)

    def __getitem__(self, idx: Union[int, str]) -> int:
        if isinstance(idx, int):
            vals = list(self._shape.values())
            if idx > len(vals):
                raise IndexError(f"Index {idx} out of range.")
            return vals[idx]
        elif isinstance(idx, str):
            if idx not in self._shape:
                raise KeyError(f"Label '{idx}' not found in layout.")
            return self._shape[idx]
        else:
            raise ValueError(
                f"Invalid type {type(idx)} for indexing. "
                "Use int or str instead."
            )

    @property
    def shape(self) -> List[int]:
        return list(self._shape.values())

    @property
    def layout_string(self) -> str:
        return "".join(self.layout)

    @staticmethod
    def _default_layout(shape: List[int]) -> List[str]:
        layout = []
        i = 0
        if shape[0] == 1:
            layout.append("N")
            i = 1
        if len(shape) - i == 3:
            if shape[i] < shape[i + 1] and shape[i] < shape[i + 2]:
                return layout + ["C", "H", "W"]
            elif shape[-1] < shape[-2] and shape[-1] < shape[-3]:
                return layout + ["H", "W", "C"]
        while len(layout) < len(shape):
            letter = chr(ord("A") + i)
            if letter not in layout:
                layout.append(letter)
            i += 1
        return layout

    def guess_new_layout(self, other: List[int]) -> "Shape":
        """Tries to guess the layout of the new shape.
        The new shape must contain the same elements as the old one.
        If two values are the same, the order of their labels will be preserved.

        Example::

            >>> shape = Shape([1, 3, 256, 256], ["N", "C", "H", "W"])
            >>> shape.guess_new_layout([1, 256, 256, 3])
            >>> Shape([1, 256, 256, 3], ["N", "H", "W", "C"])

        @type other: List[int]
        @param other: New shape to guess the layout of.

        @rtype: L{Shape}
        @return: New L{Shape} instance with guessed layout.
        """
        if len(other) != len(self):
            raise ValueError(
                "The length of the new shape must be the same as the old one."
            )
        if sorted(self._shape.values()) != sorted(other):
            raise ValueError(
                "The new shape must contain the same elements as the old one."
            )
        old_shape_tuples = list(self._shape.items())

        new_layout = []
        for dim in other:
            for i, (old_label, old_dim) in enumerate(old_shape_tuples):
                if old_dim == dim:
                    new_layout.append(old_label)
                    old_shape_tuples.pop(i)
                    break

        return Shape(other, new_layout)
