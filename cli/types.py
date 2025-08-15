import re
import click
from typing import Iterable, Any


class _BaseListType(click.ParamType):
    name = "list"

    def _split(self, raw: str) -> list[str]:
        return [p for p in re.split(r"[,\s]+", raw.strip()) if p]

    def _coerce_many(self, seq: Iterable[str]) -> list[Any]:
        raise NotImplementedError

    def convert(self, value, param, ctx):
        if isinstance(value, (list, tuple)):
            value = " ".join(map(str, value))

        s = str(value).strip()
        if not s:
            self.fail("must not be empty", param, ctx)

        parts = self._split(s)
        try:
            return self._coerce_many(parts)
        except Exception:
            self.fail(f"invalid {self.name}: {value!r}", param, ctx)


class IntList(_BaseListType):
    name = "int-list"

    def _coerce_many(self, seq: Iterable[str]) -> list[int]:
        return [int(x) for x in seq]


class StrList(_BaseListType):
    name = "str-list"

    def _coerce_many(self, seq: Iterable[str]) -> list[str]:
        return list(seq)


INT_LIST = IntList()
STR_LIST = StrList()

PARAM_TYPE_FOR_TYPE = {
    int: INT_LIST,
    str: STR_LIST,
}
