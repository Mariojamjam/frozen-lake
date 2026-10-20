"""Reading numbers typed into form fields, with a message that says what a valid value is."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    """One numeric setting: its key, its label and the values it accepts.

    `low` and `high` bound the value; `open_low` and `open_high` exclude the bound itself.
    An `optional` field may be left blank, which means "use the default" (`blank`).
    """

    key: str
    label: str
    kind: type = int
    low: float | None = None
    high: float | None = None
    open_low: bool = False
    open_high: bool = False
    optional: bool = False
    blank: str = ""

    def rule(self) -> str:
        """The accepted values, in words, for error messages."""
        if self.low is not None and self.high is not None:
            left = "(" if self.open_low else "["
            right = ")" if self.open_high else "]"
            text = f"in {left}{self.low:g}, {self.high:g}{right}"
        elif self.low is not None:
            text = f"{'above' if self.open_low else 'at least'} {self.low:g}"
        else:
            text = "a number"
        kind = "a whole number" if self.kind is int else "a number"
        return f"{kind} {text}" if text != "a number" else kind

    def parse(self, text: str) -> tuple[int | float | None, str | None]:
        """The value typed in `text`, or None and a message saying what is wrong."""
        text = text.strip()
        if not text:
            return (None, None) if self.optional else (None, f"{self.label}: required")
        try:
            value = self.kind(text)
        except ValueError:
            return None, f"{self.label}: must be {self.rule()}"
        too_low = self.low is not None and (
            value <= self.low if self.open_low else value < self.low
        )
        too_high = self.high is not None and (
            value >= self.high if self.open_high else value > self.high
        )
        if too_low or too_high:
            return None, f"{self.label}: must be {self.rule()}"
        return value, None


def show(value: int | float | None) -> str:
    """A value as it is typed into a field: blank for None, no trailing zeros."""
    if value is None:
        return ""
    return f"{value:g}" if isinstance(value, float) else str(value)
