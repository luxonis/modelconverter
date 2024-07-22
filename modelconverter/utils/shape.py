from dataclasses import dataclass
from typing import List, Union


@dataclass
class Shape(List[int]):
    """Class for handling shapes with labels.

    @type shape: List[int]
    @ivar shape: List of dimensions of the shape.

    @type layout: List[str]
    @ivar layout: List of labels for each dimension of the shape.
    """

    shape: List[int]
    layout: List[str]

    def __post_init__(self):
        self._shape = {k: v for k, v in zip(self.layout, self.shape)}

    def __getitem__(self, idx: Union[int, str]) -> int:
        if isinstance(idx, int):
            return self.shape[idx]
        elif isinstance(idx, str):
            return self._shape[idx]
        else:
            raise TypeError(
                f"Invalid type {type(idx)} for indexing. "
                "Use int or str instead."
            )

    def __len__(self) -> int:
        return len(self.shape)

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
        if sorted(self.shape) != sorted(other):
            raise ValueError(
                "The new shape must contain the same elements as the old one."
            )
        old_shape_tuples = [
            (self.shape[i], self.layout[i]) for i in range(len(self.shape))
        ]

        new_layout = []
        for dim in other:
            for i, (old_dim, old_label) in enumerate(old_shape_tuples):
                if old_dim == dim:
                    new_layout.append(old_label)
                    old_shape_tuples.pop(i)
                    break

        return Shape(other, new_layout)
