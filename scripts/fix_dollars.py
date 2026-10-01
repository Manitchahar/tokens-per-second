"""Escape currency dollar signs ($2.50 -> \\$2.50) in lesson prose so remark-math does not read them as math.

Usage: python scripts/fix_dollars.py <file.mdx> [...]
Skips frontmatter, JSX expression props (e.g. Quiz questions) and existing math that starts with a digit
is protected by writing it as $ 1/x$ (space after the dollar) in the source.
"""
import re
import sys

CURRENCY = re.compile(r'(?<![\\$])\$(?=\d)')


def fix(text: str) -> str:
    parts = text.split('\n---\n', 1)
    if len(parts) == 2 and text.startswith('---'):
        head, body = parts[0] + '\n---\n', parts[1]
    else:
        head, body = '', text
    out = []
    depth = 0  # inside a JSX {...} expression
    i = 0
    buf = ''
    while i < len(body):
        ch = body[i]
        if ch == '{':
            if depth == 0:
                out.append(CURRENCY.sub(r'\\$', buf))
                buf = ''
            depth += 1
            buf += ch
        elif ch == '}' and depth > 0:
            depth -= 1
            buf += ch
            if depth == 0:
                out.append(buf)
                buf = ''
        else:
            buf += ch
        i += 1
    out.append(CURRENCY.sub(r'\\$', buf) if depth == 0 else buf)
    return head + ''.join(out)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        with open(path, encoding='utf8') as f:
            s = f.read()
        t = fix(s)
        if t != s:
            with open(path, 'w', encoding='utf8') as f:
                f.write(t)
            print('fixed', path)
