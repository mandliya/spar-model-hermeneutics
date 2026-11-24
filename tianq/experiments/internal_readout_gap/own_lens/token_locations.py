"""The map of token locations for the best-token report: where a readout can be taken, who reads there, whether the
token scan tried it, and under each strip a real example of what sits there.

    cd tianq && python -m experiments.internal_readout_gap.own_lens.token_locations

Writes report/token_locations.png. It is a drawing, with no result file behind it:
  the scores    are copied from positions_similarity.json and positions_gap_gemma-3-27b-it__qwen3.6-27b.json (the two
                figures in figures.py)
  the examples  in A are restart point qwen36_2026-11-10_12-22-47-507049_run-18_K2 (currency conversion): the history
                from Gigi's saved run, the tokens from the saved reads under distillation_positions/. In B they are
                row 1 (an alibi conversation) from the saved reads under lie_detection_positions/
  the sources   are named in the figure's own notes

One strip per kind of input, left to right in time. A numbered pin is one location, and a Location holds its card.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from experiments.common.chart_style import GRID, INK, MUTED, SECONDARY, STYLE, SURFACE

REPORT = pathlib.Path(__file__).resolve().parents[1] / "report"
WIDTH, HEIGHT, PER_INCH = 1280, 1400, 80          # the drawing's own units; 80 of them to an inch
TRIED, OPEN, USUAL = "#1c9a8a", "#e34948", "#c98500"
HISTORY, LATE, TURN, REPLY = GRID, "#c3c2b7", "#cfe8e3", "#d3e3f6"
CARD = 172                                         # a card's height
ROW = 16                                           # the gap between two lines of example text


@dataclass
class Location:
    number: int
    tried: bool
    title: str
    who: tuple[str, ...]
    result: tuple[str, ...]
    usual: bool = False        # the token every earlier result reads


TRANSCRIPT = (
    Location(1, False, "Sentences an LLM picks",
             ("The MATS talk, for J space. An LLM reads the", "transcript and picks which sentences to read."),
             ("It needs a second model in the loop.",)),
    Location(2, False, "Late input: the last tokens of the history",
             ("Torrielli et al., for NLA. Their two best regions", "are this one and the boundary."),
             ("Cheap to add: the same run, a few more tokens.",)),
    Location(3, True, "The boundary: last token before writing",
             ("Torrielli et al. Silent Alarm. Mosafer. The MATS", "talk's NLA token, right after thinking closes."),
             ("The best one: +0.010 and +0.012 near the output.",), usual=True),
    Location(4, True, "Through Qwen3.6's turn, pasted in",
             ("John's N + 1 idea, in the version the saved", "transcripts allow."),
             ("The move is gone by 2/8. Every model is", "reading the same given words there.")),
    Location(5, False, "Inside each model's own next turn",
             ("John's N + 1 idea, the fairer version. Each", "model writes its turn and is read inside it."),
             ("It needs new model runs.",)),
    Location(6, False, "After a question asked at the end",
             ("Yuqi's idea. Ask something after the trace", "and read at the answer."),
             ("Not on transcripts. Tried on the lie", "detector, see B.")),
)
CONVERSATION = (
    Location(1, True, "Through the tested reply", ("Added here.",), ("0.61 to 0.69.",)),
    Location(2, True, "The token that ends the reply", ("Added here.",), ("0.69.",)),
    Location(3, True, "Before the answer", ("The Did-you-lie probe paper.", "Also Yuqi's idea."),
             ("The best, 0.72. The conversation's", "own words, with no model, get 0.72."), usual=True),
    Location(4, True, "The answer token", ("Added here.",), ("0.69.",)),
)
# the tokens the saved reads recorded at the eight marks, in order
TURN_TOKENS = ('" conversion"', '"Calculation"', '"Fee"', '" ("', '" Amount"', '" Conversion"', '"    "', '")"')
REPLY_TOKENS = ('" I"', '" right"', '"five"', '" quick"', '" pretty"', '" the"', '" Detective"', '"."')


class Drawing:
    def __init__(self) -> None:
        self.fig = Figure(figsize=(WIDTH / PER_INCH, HEIGHT / PER_INCH), facecolor=SURFACE)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, WIDTH)
        self.ax.set_ylim(HEIGHT, 0)
        self.ax.axis("off")

    def text(self, x, y, s, size=13, colour=INK, ha="left", bold=False, mono=False) -> None:
        self.ax.text(x, y, s, fontsize=size * 72 / PER_INCH, color=colour, ha=ha, va="baseline",
                     fontweight="bold" if bold else "normal", family="monospace" if mono else "sans-serif")

    def example(self, x, y, s, ha="center", colour=SECONDARY) -> None:
        """Real text or a real token, in typewriter type under a strip."""
        self.text(x, y, s, 10.5, colour, ha, mono=True)

    def block(self, x, y, w, h, fill, label="", edge=None, dashed=False, colour=INK) -> None:
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge or "none", linewidth=1.5 if edge else 0,
                                    linestyle=(0, (4, 3)) if dashed else "-"))
        if label:
            self.text(x + w / 2, y + h / 2 + 4.5, label, 12.5, colour, "center")

    def badge(self, x, y, number, tried, r=14) -> None:
        self.ax.add_patch(Circle((x, y), r, facecolor=TRIED if tried else SURFACE, edgecolor=TRIED if tried else OPEN,
                                 linewidth=2.4, zorder=3))
        if number:
            self.ax.text(x, y + 5, str(number), fontsize=14 * 72 / PER_INCH, color=SURFACE if tried else OPEN, ha="center",
                         fontweight="bold", zorder=4)

    def pin(self, x, y, to, loc: Location, label, ha="center", dx=0) -> None:
        self.ax.plot([x, x], [y, to], color=TRIED if loc.tried else OPEN, linewidth=1.6, zorder=2)
        self.badge(x, y, loc.number, loc.tried)
        self.text(x + dx, y - 22, label, 12.5, USUAL if loc.usual else INK, ha, bold=loc.usual)

    def marks(self, x, w, y, tokens) -> list[float]:
        """Eight marks over a block, with the token read at each in two staggered rows under the block."""
        at = [x + k / len(tokens) * w - 3 for k in range(1, len(tokens) + 1)]
        for k, (mark, token) in enumerate(zip(at, tokens)):
            self.ax.plot([mark, mark], [y - 8, y], color=TRIED, linewidth=2.2)
            self.example(mark, y + 34 + 18 + ROW * (k % 2), token, "right" if k == len(tokens) - 1 else "center")
        return at

    def card(self, x, y, w, loc: Location) -> None:
        self.ax.add_patch(Rectangle((x, y), w, CARD, facecolor="#ffffff", edgecolor=TRIED if loc.tried else OPEN, linewidth=1.4))
        self.badge(x + 26, y + 26, loc.number, loc.tried)
        self.text(x + 50, y + 31, loc.title, 14, USUAL if loc.usual else INK, bold=True)
        at = y + 56
        self.text(x + 14, at, "Who reads here", 11.5, MUTED, bold=True)
        for line in loc.who:
            at += 18
            self.text(x + 14, at, line, 13)
        at = y + 118
        self.text(x + 14, at, "Tried here", 11.5, MUTED, bold=True)
        self.text(x + 84, at, "yes" if loc.tried else "not yet", 12.5, TRIED if loc.tried else OPEN, bold=True)
        for line in loc.result:
            at += 18
            self.text(x + 14, at, line, 13)


def main() -> None:
    with rc_context(STYLE):
        d = Drawing()
        d.text(30, 36, "Where can J space be read? Every token location, who reads there, and whether the token scan tried it", 19)
        d.badge(44, 64, "", True, 8)
        d.text(60, 69, "tried in the token scan", 13)
        d.badge(250, 64, "", False, 8)
        d.text(266, 69, "not tried yet", 13)
        d.text(390, 69, "amber name = the token every earlier result reads", 13, USUAL, bold=True)
        d.example(800, 69, "typewriter text under a strip = a real example", "left")

        # A. an agent transcript at a restart point
        d.text(30, 110, "A. An agent transcript at a restart point (the distillation)", 16, SECONDARY, bold=True)
        sy = 196
        first = sy + 34 + 18                                   # the first line of example text under the strip
        d.block(44, sy, 286, 34, HISTORY, "the history so far")
        d.block(330, sy, 100, 34, LATE, "its last tokens")
        d.block(434, sy, 28, 34, USUAL)
        d.block(466, sy, 520, 34, TURN, "the next turn: Qwen3.6's actual words, pasted in")
        d.block(990, sy, 180, 34, HISTORY, "a question is asked")
        d.block(1174, sy, 28, 34, LATE)
        a = {loc.number: loc for loc in TRANSCRIPT}
        d.ax.plot([90, 90, 290, 290], [sy, sy - 16, sy - 16, sy], color=OPEN, linewidth=1.6)
        d.ax.plot([190, 190], [sy - 16, sy], color=OPEN, linewidth=1.6)
        d.pin(190, sy - 44, sy - 16, a[1], "sentences an LLM picks")
        d.pin(380, sy - 44, sy, a[2], "late input", ha="right", dx=14)
        d.pin(448, sy - 44, sy, a[3], "the boundary token", ha="left", dx=-14)
        marks = d.marks(466, 520, sy, TURN_TOKENS)
        d.pin(marks[4], sy - 44, sy - 8, a[4], "8 tokens through the pasted turn")
        d.pin(1188, sy - 44, sy, a[6], "after a question")
        d.example(44, first, "user: Convert 1250.0 USD to INR using a Bank", "left")
        d.example(44, first + ROW, "Transfer. Show me the precise calculation.", "left")
        d.example(44, first + 2 * ROW, 'tool: {"status": "success", "rate": 83.58}', "left")
        d.example(380, first, "..._amount}')")
        d.example(448, first, '"\\n"')
        d.example(466, first + 2 * ROW, "Here is the detailed breakdown of the currency conversion: ...", "left")
        d.example(1080, first, "not run, so no example", colour=MUTED)
        own = sy + 112                                         # the model's own next turn, under the pasted one
        d.block(466, own, 520, 34, SURFACE, "or: the model's own next turn, which it writes itself", edge=TRIED, dashed=True, colour=SECONDARY)
        d.ax.plot([424, 466], [own + 17, own + 17], color=OPEN, linewidth=1.6, zorder=2)
        d.badge(410, own + 17, a[5].number, False)
        d.text(388, own + 22, "inside the model's own turn", 12.5, INK, "right")
        d.example(466, own + 52, 'not run. Likeliest first word: Qwen3.6 "Here", base Gemma "Here", distilled Gemma "```"', "left", MUTED)
        top = own + 76
        for k, loc in enumerate(TRANSCRIPT):
            d.card(30 + 412 * (k % 3), top + (CARD + 12) * (k // 3), 396, loc)

        # B. a lie detection conversation
        top += 2 * (CARD + 12) + 36
        d.text(30, top, "B. A lie detection conversation (the lie detector)", 16, SECONDARY, bold=True)
        sy = top + 76
        first = sy + 34 + 18
        d.block(44, sy, 250, 34, HISTORY, "the conversation so far")
        d.block(298, sy, 470, 34, REPLY, "the tested reply, in Gemma-3-27B's own words")
        d.block(772, sy, 28, 34, LATE)
        d.block(804, sy, 250, 34, HISTORY, "user: Did you lie?")
        d.block(1058, sy, 28, 34, USUAL)
        d.block(1090, sy, 28, 34, LATE)
        b = {loc.number: loc for loc in CONVERSATION}
        marks = d.marks(298, 470, sy, REPLY_TOKENS)
        d.pin(marks[3], sy - 40, sy - 8, b[1], "8 tokens through the reply")
        d.pin(786, sy - 40, sy, b[2], "end of reply")
        d.pin(1072, sy - 40, sy, b[3], "before the answer", ha="right", dx=16)
        d.pin(1104, sy - 40, sy, b[4], "the answer token", ha="left", dx=-12)
        d.example(169, first, "an alibi conversation", colour=MUTED)
        d.example(786, first + 2 * ROW, '"<end_of_turn>"')
        d.example(1072, first, '"\\n"')
        d.example(1104, first, '"A"')
        top = first + 2 * ROW + 22
        for k, loc in enumerate(CONVERSATION):
            d.card(30 + 308 * k, top, 296, loc)

        notes = ("How to read this",
                 "• Each strip is one input to the model, left to right in time. The model starts writing at the amber token.",
                 "• A numbered pin is one token location where J space can be read. The card with the same number says who reads there and whether the token scan tried it.",
                 "• Filled green: read in the token scan. Red ring: not read yet. Amber: the token every earlier result reads, the last one before the model writes.",
                 "• Typewriter text under a strip is real: in A one restart point of the currency conversion task, as base Gemma reads it; in B one alibi conversation.",
                 "• Under a row of eight marks, the token read at each mark sits below it, on two staggered lines.",
                 "• The boundary token is a formatting token: for Gemma the line break after <start_of_turn>model, for Qwen3.6 the blank line after </think>.",
                 "• A is the distillation: Qwen3.6, base Gemma and the distilled Gemma at the 327 restart points. The numbers are the change in same − chance, distilled Gemma minus base Gemma.",
                 "• B is the lie detector: Gemma-3-27B's 2,776 conversations. The numbers are the target's AUROC with a classifier on its J-space words, for the kind of lie it was trained on; 0.5 is chance.",
                 "• Sources: the MATS talk is Xing and Lin, \"How good are NLA and J-Space for monitoring?\". Torrielli et al. is arXiv 2609.37040. Silent Alarm and Mosafer are the two J-space papers that compare models.")
        top += CARD + 34
        for k, line in enumerate(notes):
            d.text(30, top + 21 * k, line, 11.5, MUTED)
    path = REPORT / "token_locations.png"
    d.fig.savefig(path, dpi=200)
    print(f"wrote {path}; the notes end at {top + 21 * len(notes)} of {HEIGHT}", flush=True)


if __name__ == "__main__":
    main()
