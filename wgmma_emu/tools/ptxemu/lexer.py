"""Statement-level lexer for PTX text.

The lexer splits a PTX module into a flat list of elements:

* ``comment``          ``// ...`` or ``/* ... */`` (outside statements)
* ``asm_begin``/``asm_end``  the ``// begin inline asm`` / ``// end inline asm`` markers cicc
  writes around every inline-asm expansion; elements in between carry ``in_asm=True``
* ``label``            ``name:`` at statement start (text is the bare name)
* ``lbrace``/``rbrace``  block braces at statement start (function bodies, inline-asm scopes,
  callseq blocks)
* ``line_directive``   newline-terminated directives (``.version``, ``.target``, ``.loc``,
  ``.file``, performance-tuning and cluster directives, ``@@DWARF``)
* ``block_directive``  brace-delimited module directives (``.section``,
  ``.metadata_section``, ``.nv_intermediate_source_section``), kept verbatim
* ``header``           a ``.entry``/``.func`` header that is followed by a body; the ``{`` is
  emitted as the next ``lbrace`` element
* ``statement``        any ``;``-terminated statement (instruction or directive). The text
  excludes the ``;``; comments inside the statement are replaced by a space and emitted as
  separate ``comment`` elements right after the statement.

Brace, bracket and parenthesis nesting inside a statement (vector operands ``{%r1, %r2}``,
addresses ``[%rd1+16]``, tensor addresses ``[tmap, {%r1, %r2}]``, initializers ``= {1, 2}``,
call argument lists) is tracked, so a statement may span several lines. Strings
(``.pragma "nounroll"``, ``.file 1 "a.cu"``) are skipped as a unit.

Unknown or malformed input raises :class:`LexError`; nothing is silently dropped.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

COMMENT = "comment"
ASM_BEGIN = "asm_begin"
ASM_END = "asm_end"
LABEL = "label"
LBRACE = "lbrace"
RBRACE = "rbrace"
LINE_DIRECTIVE = "line_directive"
BLOCK_DIRECTIVE = "block_directive"
HEADER = "header"
STATEMENT = "statement"

# Directives that are terminated by the end of the line (no ';').
LINE_DIRECTIVES = frozenset(
    {
        ".version",
        ".target",
        ".address_size",
        ".file",
        ".loc",
        ".loc_intermediate",
        ".maxnreg",
        ".maxntid",
        ".reqntid",
        ".minnctapersm",
        ".maxnctapersm",
        ".noreturn",
        ".explicitcluster",
        ".reqnctapercluster",
        ".maxclusterrank",
        ".blocksareclusters",
        ".language",
        ".abi_preserve",
        ".abi_preserve_control",
        ".minperctamemory",
    }
)

# Module directives whose payload is a brace-delimited block without ';' terminators.
BLOCK_DIRECTIVES = frozenset(
    {".section", ".metadata_section", ".nv_intermediate_source_section"}
)

_WS_RE = re.compile(r"\s+")
_LABEL_RE = re.compile(r"([A-Za-z_$][\w$]*)[ \t]*:(?!:)")
_DIRECTIVE_WORD_RE = re.compile(r"\.[A-Za-z_]\w*")
_STRING_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"')
_STMT_SPECIAL_RE = re.compile(r'["/(){}\[\];]')
_LINE_SPECIAL_RE = re.compile(r'["/\n]')
_BLOCK_SPECIAL_RE = re.compile(r'["{}/]|\.source_begin\b')
_HEADER_RE = re.compile(
    r"(?:\.(?:visible|extern|weak|common)\s+)*\.(?:entry|func)(?![\w$])"
)
_ASM_BEGIN_RE = re.compile(r"//\s*begin inline asm\s*$")
_ASM_END_RE = re.compile(r"//\s*end inline asm\s*$")


class LexError(ValueError):
    """Malformed PTX text."""

    def __init__(self, msg: str, lineno: int):
        super().__init__(f"line {lineno}: {msg}")
        self.lineno = lineno


@dataclass
class Element:
    kind: str
    text: str
    lineno: int
    in_asm: bool = False
    start: int = 0  # offset of the first character in the source
    end: int = 0  # offset one past the last character (includes ';' for statements)


def is_header_text(text: str) -> bool:
    """True if ``text`` starts like a ``.entry``/``.func`` header or prototype."""
    return _HEADER_RE.match(text.lstrip()) is not None


class _Lexer:
    def __init__(self, src: str):
        self.src = src
        self.n = len(src)
        self.newlines = [m.start() for m in re.finditer("\n", src)]
        self.out: list[Element] = []
        self.in_asm = False

    # -- helpers ---------------------------------------------------------------------------
    def lineno(self, pos: int) -> int:
        return bisect.bisect_left(self.newlines, pos) + 1

    def emit(self, kind: str, text: str, start: int, end: int) -> None:
        self.out.append(
            Element(kind, text, self.lineno(start), self.in_asm, start, end)
        )

    def skip_string(self, pos: int) -> int:
        m = _STRING_RE.match(self.src, pos)
        if m is None:
            raise LexError("unterminated string literal", self.lineno(pos))
        return m.end()

    def skip_comment(self, pos: int) -> int:
        """``pos`` points at '/', which starts a comment. Returns the offset after it."""
        src = self.src
        if src.startswith("//", pos):
            k = src.find("\n", pos)
            return self.n if k < 0 else k
        k = src.find("*/", pos + 2)
        if k < 0:
            raise LexError("unterminated /* comment", self.lineno(pos))
        return k + 2

    def next_significant(self, pos: int) -> int:
        """Offset of the next character that is neither whitespace nor inside a comment."""
        src = self.src
        while pos < self.n:
            m = _WS_RE.match(src, pos)
            if m:
                pos = m.end()
                continue
            if src.startswith("//", pos) or src.startswith("/*", pos):
                pos = self.skip_comment(pos)
                continue
            break
        return pos

    def emit_comment(self, start: int, end: int) -> None:
        text = self.src[start:end].rstrip()
        if _ASM_BEGIN_RE.match(text):
            self.in_asm = True
            self.emit(ASM_BEGIN, text, start, end)
        elif _ASM_END_RE.match(text):
            self.emit(ASM_END, text, start, end)
            self.in_asm = False
        else:
            self.emit(COMMENT, text, start, end)

    # -- main loop -------------------------------------------------------------------------
    def run(self) -> list[Element]:
        src, n = self.src, self.n
        pos = 0
        while True:
            m = _WS_RE.match(src, pos)
            if m:
                pos = m.end()
            if pos >= n:
                break
            c = src[pos]
            if c == "/" and (src.startswith("//", pos) or src.startswith("/*", pos)):
                end = self.skip_comment(pos)
                self.emit_comment(pos, end)
                pos = end
                continue
            if c == "{":
                self.emit(LBRACE, "{", pos, pos + 1)
                pos += 1
                continue
            if c == "}":
                self.emit(RBRACE, "}", pos, pos + 1)
                pos += 1
                continue
            if c == ".":
                m = _DIRECTIVE_WORD_RE.match(src, pos)
                word = m.group(0) if m else ""
                if word in LINE_DIRECTIVES:
                    pos = self.line_directive(pos)
                    continue
                if word in BLOCK_DIRECTIVES:
                    pos = self.block_directive(pos, word)
                    continue
            elif c == "@":
                if src.startswith("@@", pos):
                    pos = self.line_directive(pos)
                    continue
            else:
                m = _LABEL_RE.match(src, pos)
                if m:
                    self.emit(LABEL, m.group(1), pos, m.end())
                    pos = m.end()
                    continue
            pos = self.statement(pos)
        return self.out

    def line_directive(self, pos: int) -> int:
        src, n = self.src, self.n
        i = pos
        while True:
            m = _LINE_SPECIAL_RE.search(src, i)
            if m is None:
                end = n
                break
            j = m.start()
            ch = src[j]
            if ch == '"':
                i = self.skip_string(j)
                continue
            if ch == "/":
                if src.startswith("//", j) or src.startswith("/*", j):
                    end = j
                    break
                i = j + 1
                continue
            end = j  # newline
            break
        text = src[pos:end].rstrip()
        self.emit(LINE_DIRECTIVE, text, pos, pos + len(text))
        return end

    def block_directive(self, pos: int, word: str) -> int:
        src = self.src
        i = pos + len(word)
        depth = 0
        while True:
            m = _BLOCK_SPECIAL_RE.search(src, i)
            if m is None:
                raise LexError(f"unterminated {word} block", self.lineno(pos))
            j = m.start()
            tok = m.group(0)
            if tok == '"':
                i = self.skip_string(j)
                continue
            if tok == "/":
                if src.startswith("//", j) or src.startswith("/*", j):
                    i = self.skip_comment(j)
                else:
                    i = j + 1
                continue
            if tok == ".source_begin":
                k = src.find(".source_end", j)
                if k < 0:
                    raise LexError(".source_begin without .source_end", self.lineno(j))
                i = k + len(".source_end")
                continue
            if tok == "{":
                depth += 1
                i = j + 1
                continue
            depth -= 1  # '}'
            if depth < 0:
                raise LexError(f"unbalanced '}}' in {word} block", self.lineno(j))
            i = j + 1
            if depth == 0:
                break
        self.emit(BLOCK_DIRECTIVE, src[pos:i], pos, i)
        return i

    def statement(self, pos: int) -> int:
        src, n = self.src, self.n
        i = pos
        seg = pos
        depth = 0
        pieces: list[str] = []
        comments: list[tuple[int, int]] = []

        def current(j: int) -> str:
            return ("".join(pieces) + src[seg:j]).strip()

        while True:
            m = _STMT_SPECIAL_RE.search(src, i)
            if m is None:
                raise LexError("unterminated statement (missing ';')", self.lineno(pos))
            j = m.start()
            ch = src[j]
            if ch == '"':
                i = self.skip_string(j)
                continue
            if ch == "/":
                if src.startswith("//", j) or src.startswith("/*", j):
                    k = self.skip_comment(j)
                    pieces.append(src[seg:j])
                    pieces.append(" ")
                    comments.append((j, k))
                    i = seg = k
                else:
                    i = j + 1
                continue
            if ch in "([":
                depth += 1
                i = j + 1
                continue
            if ch == "{":
                if depth == 0 and is_header_text(current(j)):
                    text = current(j)
                    self.emit(HEADER, text, pos, j)
                    for cs, ce in comments:
                        self.emit_comment(cs, ce)
                    return j
                depth += 1
                i = j + 1
                continue
            if ch in ")]}":
                if depth == 0:
                    raise LexError(f"unbalanced '{ch}'", self.lineno(j))
                depth -= 1
                i = j + 1
                continue
            # ch == ';'
            if depth != 0:
                raise LexError("';' inside an unclosed bracket", self.lineno(j))
            text = current(j)
            if is_header_text(text):
                k = self.next_significant(j + 1)
                if k < n and src[k] == "{":
                    # '.entry f .minperctamemory 2048; { ... }': the ';' belongs to the header.
                    i = j + 1
                    continue
            if text:  # a bare ';' is an empty statement with no effect
                self.emit(STATEMENT, text, pos, j + 1)
            for cs, ce in comments:
                self.emit_comment(cs, ce)
            return j + 1


def lex(src: str) -> list[Element]:
    """Split PTX source text into :class:`Element` objects (see module docstring)."""
    return _Lexer(src).run()
