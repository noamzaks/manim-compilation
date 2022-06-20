from manim import *
from .utilities import *
from typing import Tuple


Token = str
Word = List[Token]


class Transition:
    def __init__(self, index: int, nonterminal: Token, result: Word):
        self.index = index
        self.nonterminal = nonterminal
        self.result = result

    def __repr__(self) -> str:
        return f"{self.nonterminal} -> {''.join(self.result)}"


class Derivations:
    """A class to represent a series of derivations"""

    derivations: List[Word]
    rules: List[Tuple[Word, Word]]
    transitions: List[Transition]

    def __init__(self, derivations: List[Word], rules: List[Tuple[Word, Word]]):
        self.derivations = derivations
        self.rules = rules
        self.nonterminals = {rule[0] for rule in rules}
        self.transitions = []

        for i in range(1, len(derivations)):
            previous = derivations[i - 1]
            current = derivations[i]

            found = False
            for i in range(len(previous)):
                if previous[i] not in self.nonterminals:
                    continue

                for rule in self.rules:
                    if rule[0] != previous[i]:
                        continue

                    if previous[:i] + rule[1] + previous[i + 1 :] == current:
                        self.transitions.append(Transition(i, rule[0], rule[1]))
                        found = True
                        break

                if found:
                    break
            if not found:
                raise Exception(
                    "Could not find derivation from "
                    + "".join(previous)
                    + " to "
                    + "".join(current)
                )


class DerivationColumn(VGroup):
    """Represents a step-by-step derivation in a column of words"""

    def __init__(self, derivations: Derivations, show_tree_edges=False):
        self.derivations = derivations
        self.index = 0
        self.show_tree_edges = show_tree_edges
        self.dots_in_transition = []
        self.edges_in_transition = []
        vmobjects = [Tex(*derivation) for derivation in derivations.derivations]

        VGroup.__init__(self, *vmobjects)
        self.arrange(DOWN, center=False)

    def __getitem__(self, value) -> Tex:
        return super().__getitem__(value)

    def highlight_current(self):
        """Highlights the current nonterminal about to transition using a production"""
        transition = self.derivations.transitions[self.index]

        return (
            self[self.index]
            .get_part_by_tex(transition.nonterminal)
            .animate.set_color(YELLOW)
        )

    def transition(self):
        """Applies the next production"""
        transition = self.derivations.transitions[self.index]

        new_dot = self[self.index].submobjects[transition.index].copy()
        self.dots_in_transition.append(
            [
                mob.copy()
                for mob in self[self.index + 1].submobjects[
                    transition.index : transition.index + len(transition.result)
                ]
            ]
        )
        self.edges_in_transition.append(
            [
                Line(
                    self[self.index]
                    .submobjects[transition.index]
                    .get_boundary_point(DOWN)[1]
                    * UP
                    + self[self.index].submobjects[transition.index].get_center()[0]
                    * RIGHT,
                    mobject.get_boundary_point(UP)[1] * UP
                    + mobject.get_center()[0] * RIGHT,
                )
                for mobject in self[self.index + 1].submobjects[
                    transition.index : transition.index + len(transition.result)
                ]
            ]
        )
        animation = AnimationGroup(
            self[self.index].submobjects[transition.index].animate.set_color(WHITE),
            Transform(
                VGroup(*self[self.index].submobjects[: transition.index]),
                VGroup(*self[self.index + 1].submobjects[: transition.index]),
            ),
            ReplacementTransform(
                new_dot,
                VGroup(
                    *self[self.index + 1].submobjects[
                        transition.index : transition.index + len(transition.result)
                    ]
                ),
            ),
            Transform(
                VGroup(*self[self.index].submobjects[transition.index + 1 :]),
                VGroup(
                    *self[self.index + 1].submobjects[
                        transition.index + len(transition.result) :
                    ]
                ),
            ),
            *[
                Create(line)
                for line in self.edges_in_transition[-1]
                if self.show_tree_edges
            ],
        )
        self.index += 1
        return animation, self.dots_in_transition[-1]

    def get_transform_to_cst(self):
        cst = ConcreteSyntaxTree(self.derivations).move_to(self)
        return AnimationGroup(
            *[FadeOut(mobject) for mobject in self if mobject != self[0]],
            Transform(self[0], cst.root),
            FadeIn(cst.root.submobjects[0]),
            *[
                Transform(dot, cst_dot)
                for dot, cst_dot in zip(
                    flatten(self.dots_in_transition), flatten(cst.dots_in_transition)
                )
            ],
            *[FadeIn(dot.submobjects[0]) for dot in flatten(cst.dots_in_transition)],
            *[
                Transform(edge, cst_edge)
                for edge, cst_edge in zip(
                    flatten(self.edges_in_transition), flatten(cst.edges_in_transition)
                )
            ],
        )

    @property
    def finished(self):
        return self.index == len(self.derivations.transitions)

    def get_transition(self):
        return self.derivations.transitions[self.index]


class ConcreteSyntaxTree(VGroup):
    def __init__(
        self,
        derivations: Derivations,
        radius=0.3,
        buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
    ):
        vmobjects = []
        self.index = 0

        for token in derivations.derivations[-1]:
            vmobjects.append(LabeledDot(token, radius=radius))

        self.columns = list(map(lambda mob: [mob], vmobjects))
        self.active_columns = self.columns.copy()
        self.dots_in_transition = []
        self.edges_in_transition = []
        self.highlight_in_transition = []

        VGroup.__init__(self, *vmobjects)
        self.arrange(RIGHT, center=False, buff=buff)

        for transition in derivations.transitions[::-1]:
            self.dots_in_transition.append([])
            self.edges_in_transition.append([])
            dot = LabeledDot(transition.nonterminal, radius=radius)
            dot.next_to(self.columns[transition.index][-1], UP, buff=buff)
            for i in range(1, len(transition.result)):
                if (
                    dot.get_top()[1]
                    < self.active_columns[transition.index + i][-1].get_top()[1]
                ):
                    dot.next_to(
                        self.active_columns[transition.index + i][-1], UP, buff=buff
                    )
            dot.set_x(
                (
                    self.active_columns[transition.index][-1].get_x()
                    + self.active_columns[
                        transition.index + len(transition.result) - 1
                    ][-1].get_x()
                )
                / 2
            )

            self.highlight_in_transition.append(dot)
            self.add(dot)
            line = Line(dot, self.active_columns[transition.index][-1])
            self.add(line)
            self.dots_in_transition[-1].append(
                self.active_columns[transition.index][-1]
            )
            self.edges_in_transition[-1].append(line)
            for i in range(1, len(transition.result)):
                line = Line(dot, self.active_columns[transition.index + 1][-1])
                self.add(line)
                self.edges_in_transition[-1].append(line)
                self.dots_in_transition[-1].append(
                    self.active_columns[transition.index + 1][-1]
                )
                self.active_columns.pop(transition.index + 1)
            self.columns[transition.index].append(dot)

            self.root = dot
        self.dots_in_transition.reverse()
        self.edges_in_transition.reverse()
        self.highlight_in_transition.reverse()

    def transition(self):
        animation = AnimationGroup(
            self.highlight_in_transition[self.index].animate.set_color(WHITE),
            *[FadeIn(dot) for dot in self.dots_in_transition[self.index]],
            *[FadeIn(line) for line in self.edges_in_transition[self.index]],
        )
        self.index += 1
        return animation

    def highlight_current(self):
        return AnimationGroup(
            self.highlight_in_transition[self.index].animate.set_color(YELLOW),
        )

    @property
    def finished(self):
        return self.index == len(self.derivations.transitions)


class GrammarRules(VGroup):
    def __init__(self, derivations: Derivations):
        vmobjects = [Tex(bold("Grammar"))]
        self.map = {}
        for nonterminal in derivations.nonterminals:
            results = []
            for rule in derivations.rules:
                if rule[0] != nonterminal:
                    continue
                results.append("{{ " + "".join(rule[1]) + " }}")
            tex = Tex(
                "{{" + nonterminal + "}} $\\longrightarrow$ " + " | ".join(results)
            )
            vmobjects.append(tex)
            self.map[nonterminal] = tex
        VGroup.__init__(self, *vmobjects)

        self.arrange(DOWN, center=False, aligned_edge=LEFT)

    def highlight_transition(self, transition: Transition):
        return AnimationGroup(
            self.map[transition.nonterminal].submobjects[0].animate.set_color(YELLOW),
            self.map[transition.nonterminal]
            .get_part_by_tex("$\\longrightarrow$")
            .animate.set_color(YELLOW),
            self.map[transition.nonterminal]
            .get_part_by_tex("".join(transition.result))
            .animate.set_color(YELLOW),
        )

    def clear(self):
        return AnimationGroup(*[mobject.animate.set_color(WHITE) for mobject in self])


class LookaheadIndicator(VGroup):
    def __init__(self, derivations: Derivations):
        self.derivations = derivations
        self.last = self.derivations.derivations[-1]
        self.index = 0
        self.lookahead_index = 0

        self.tex = Tex(*self.last)
        self.arrow = self._get_arrow()

        VGroup.__init__(self, self.tex, self.arrow)

    def _get_arrow(self):
        edge_point = (
            self.tex[self.lookahead_index].get_center()[0] * RIGHT
            + self.tex.get_boundary_point(UP)[1] * UP
        )
        return Arrow(edge_point + UP, edge_point)

    def highlight_current(self):
        return AnimationGroup(
            self.tex[self.lookahead_index].animate.set_color(YELLOW),
        )

    def _get_next_lookahead_index(self):
        current = self.derivations.derivations[self.index + 1]
        lookahead_index = 0
        while (
            lookahead_index < len(current) - 1
            and current[lookahead_index] == self.last[lookahead_index]
        ):
            lookahead_index += 1
        return lookahead_index

    def transition(self):
        original_lookahead_index = self.lookahead_index
        self.lookahead_index = self._get_next_lookahead_index()
        animation = AnimationGroup(
            self.tex[original_lookahead_index].animate.set_color(WHITE),
            Transform(self.arrow, self._get_arrow()),
        )
        self.index += 1
        return animation
