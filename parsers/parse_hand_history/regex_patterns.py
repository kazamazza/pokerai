# ── CONFIG ─────────────────────────────────────────────────────────────────────
import re

HAND_SPLIT_RE = re.compile(r"(?=^PokerStars Hand #\d+)", re.MULTILINE)
HOLE_CARDS_RE = re.compile(r"Dealt to (.+?) \[([2-9TJQKA][cdhs]) ([2-9TJQKA][cdhs])\]")
SHOWDOWN_RE = re.compile(r"^(.+?): shows \[([2-9TJQKA][cdhs]) ([2-9TJQKA][cdhs])\]")
ACTION_LINE_RE = re.compile(r"^(.+?): (.+)$")
TABLE_RE = re.compile(r"Table '(.+?)' \d+-max Seat #(\d+) is the button")
SEATLINE_RE = re.compile(r"^Seat (\d+): (\S+)(?: \(\$(\d+\.?\d*) in chips\))")
FLOP_RE = re.compile(r"\*\*\* FLOP \*\*\* \[([^\]]+)\]")
TURN_RE = re.compile(r"\*\*\* TURN \*\*\* \[[^\]]+\] \[([^\]]+)\]")
RIVER_RE = re.compile(r"\*\*\* RIVER \*\*\* \[[^\]]+\] \[([^\]]+)\]")
FOLD_RE = re.compile(r"^(.+?) folds")
SUMMARY_FOLD_RE = re.compile(r"Seat \d+ (.+?) \(.+\) folded before (Flop|Turn|River)")