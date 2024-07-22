from typing import List, Union
from dataclasses import dataclass, Field


@dataclass
class Shape:
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

        Example::

            >>> shape = Shape([1, 3, 256, 256], ["N", "C", "H", "W"])
            >>> shape.guess_new_layout([3, 256, 256, 1])
            >>> Shape([1, 256, 256, 3], ["N", "H", "W", "C"])

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
