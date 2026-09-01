"""Shared ``hypothesis`` strategies for the host-side unit tests.

Deliberately small (few dimensions, short names, tiny images): every property
built on them is about bookkeeping -- shapes, layouts, key sets, file counts --
so larger values would only buy runtime.
"""

import string

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from luxonis_ml.typing import Params

# ``work_dir`` and ``monkeypatch`` are function-scoped, so Hypothesis hands the
# same one to every example of a test instead of a fresh one. A handful of
# tests want exactly that -- each example overwrites the files it reads back --
# and mark themselves with this; anywhere else the health check still fires.
reuses_function_fixtures = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)

# ``make_default_layout`` labels dimensions from a 26-letter alphabet, so
# ranks beyond that have no layout to produce. Stopping well short keeps
# the generated shapes readable in failure output.
MAX_RANK = 16

dim_sizes = st.integers(min_value=1, max_value=64)

tensor_names = st.text(
    alphabet=string.ascii_letters + string.digits + "_./",
    min_size=1,
    max_size=12,
)

# Small enough to resize quickly, large enough that a padded or cropped
# region is several pixels across in both directions.
image_sides = st.integers(min_value=8, max_value=64)
image_sizes = st.tuples(image_sides, image_sides)


def shapes(
    min_rank: int = 1, max_rank: int = MAX_RANK
) -> st.SearchStrategy[list[int]]:
    """Tensor shapes of rank ``min_rank..max_rank``."""
    return st.lists(dim_sizes, min_size=min_rank, max_size=max_rank)


@st.composite
def shape_permutations(
    draw: st.DrawFn, min_rank: int = 1, max_rank: int = MAX_RANK
) -> tuple[list[int], list[int]]:
    """A shape together with a permutation of it.

    Drawing the pair as a unit is what makes ``guess_new_layout``
    testable: it rejects any new shape that is not a rearrangement of
    the old one, so two independent shape draws would be filtered away.
    """
    shape = draw(shapes(min_rank, max_rank))
    order = draw(st.permutations(range(len(shape))))
    return shape, [shape[i] for i in order]


# The four fields ``_expand_encoding_item`` broadcasts over channels.
VECTOR_ENCODING_KEYS = ("scale", "offset", "min", "max")

_finite_floats = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
)
_offsets = st.integers(min_value=-(2**16), max_value=2**16)

_ENCODING_FIELDS = {
    "scale": _finite_floats,
    "offset": _offsets,
    "min": _finite_floats,
    "max": _finite_floats,
}

# Scalar fields, in draw order. The trailing four are what
# ``_normalize_encoding_item`` is expected to fold away: the two short aliases
# and anything outside ``ALLOWED_ENCODING_KEYS``.
_SCALAR_FIELDS = {
    "bitwidth": st.sampled_from([4, 8, 16, 32]),
    "is_symmetric": st.booleans(),
    "dtype": st.sampled_from(["int", "INT", "float", "Float"]),
    "bw": st.sampled_from([4, 8, 16, 32]),
    "is_sym": st.booleans(),
    "name": tensor_names,
    "encoding_type": st.sampled_from(["PER_TENSOR", "PER_CHANNEL"]),
}


@st.composite
def encoding_items(
    draw: st.DrawFn, *, vector_length: int | None = None
) -> Params:
    """A single raw quantization-encoding entry.

    Values are valid for ``QuantizationOverridesItem``, so one strategy drives
    both the private normalisation helpers and the public ``parse_encodings``.
    With ``vector_length``, every per-channel field is drawn as a list of that
    length -- mismatched lengths are a rejected input with its own test.
    """
    item: Params = {}
    for key, values in _ENCODING_FIELDS.items():
        if not draw(st.booleans()):
            continue
        item[key] = (
            draw(
                st.lists(
                    values, min_size=vector_length, max_size=vector_length
                )
            )
            if vector_length is not None
            else draw(values)
        )

    for key, values in _SCALAR_FIELDS.items():
        if draw(st.booleans()):
            item[key] = draw(values)
    return item


@st.composite
def encoding_groups(draw: st.DrawFn) -> dict[str, list[Params]]:
    """A ``{tensor_name: [entry, ...]}`` encoding group."""
    names = draw(st.lists(tensor_names, min_size=1, max_size=3, unique=True))
    return {
        name: draw(
            st.lists(
                st.integers(min_value=0, max_value=4).flatmap(
                    lambda n: encoding_items(vector_length=n or None)
                ),
                min_size=1,
                max_size=2,
            )
        )
        for name in names
    }
