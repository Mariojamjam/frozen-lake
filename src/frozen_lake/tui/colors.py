"""The "Frost" palette of the terminal explorer: every color the interface uses, in one place.

Night-violet surfaces, lavender ice, an aurora-teal goal and rose for danger. To restyle the
interface, change the values here; the app and its stylesheet only refer to these names.
"""

from textual.theme import Theme

# Surfaces
NIGHT = "#0d0a1c"  # screen background
SURFACE = "#161229"  # panels
PANEL = "#221b3d"  # borders of secondary panels

# Violet family
VIOLET = "#8b5cf6"  # primary: lake border, start tile, policy arrows
LAVENDER = "#a78bfa"  # secondary borders
LILAC = "#c4b5fd"  # supporting text
ORCHID = "#e879f9"  # accent: the agent, the chosen action, the previous tile

# Lake tiles
ICE = ("#ddd6fe", "#cfc4fc")  # checkered so neighboring tiles stay distinguishable
ICE_MARK = "#8b7fd6"  # the dot drawn on frozen tiles
DEEP_WATER = "#140d33"  # holes: dark water under the ice
AURORA = "#2dd4bf"  # the goal

# Text and signals
SNOW = "#ffffff"  # letters on dark tiles
FROST_TEXT = "#ece8ff"  # default text
MINT = "#5eead4"  # a move that went as intended
AMBER = "#fbbf24"  # a slip, a timeout
ROSE = "#fb7185"  # danger: a hole, a fall

FROST_THEME = Theme(
    name="frost",
    primary=VIOLET,
    secondary=LAVENDER,
    accent=ORCHID,
    warning=AMBER,
    error=ROSE,
    success=AURORA,
    foreground=FROST_TEXT,
    background=NIGHT,
    surface=SURFACE,
    panel=PANEL,
    dark=True,
)
