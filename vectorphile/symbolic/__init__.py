"""vectorphile.symbolic — Symbolic mathematics (SymPy-compatible subset).

Provides basic symbolic expression manipulation: variables, arithmetic,
differentiation, simplification, substitution, and LaTeX output.

No external dependencies.

Usage::

    from vectorphile.symbolic import Symbol, symbols, diff, integrate, expand, simplify

    x, y = symbols('x y')
    expr = x**2 + 2*x*y + y**2
    d = diff(expr, x)              # 2*x + 2*y
    simplified = simplify(expr)    # (x + y)^2 (if recognized)
"""
from __future__ import annotations
import math
import re
from typing import Dict, Optional, Union


# ---------------------------------------------------------------------------
# Expression base
# ---------------------------------------------------------------------------

class Expr:
    """Base class for symbolic expressions."""

    def __add__(self, other):
        return Add(self, _to_expr(other))

    def __radd__(self, other):
        return Add(_to_expr(other), self)

    def __sub__(self, other):
        return Sub(self, _to_expr(other))

    def __rsub__(self, other):
        return Sub(_to_expr(other), self)

    def __mul__(self, other):
        return Mul(self, _to_expr(other))

    def __rmul__(self, other):
        return Mul(_to_expr(other), self)

    def __truediv__(self, other):
        return Div(self, _to_expr(other))

    def __rtruediv__(self, other):
        return Div(_to_expr(other), self)

    def __pow__(self, other):
        return Pow(self, _to_expr(other))

    def __rpow__(self, other):
        return Pow(_to_expr(other), self)

    def __neg__(self):
        return Mul(Number(-1), self)

    def __eq__(self, other):
        return Eq(self, _to_expr(other))

    def __repr__(self):
        return self.__str__()

    def subs(self, replacements: dict) -> 'Expr':
        """Substitute variables with values or expressions."""
        raise NotImplementedError

    def diff(self, var: 'Symbol', n: int = 1) -> 'Expr':
        """Differentiate n times with respect to var."""
        result = self
        for _ in range(n):
            result = result._diff(var)
        return simplify(result)

    def _diff(self, var: 'Symbol') -> 'Expr':
        raise NotImplementedError

    def eval(self, **kwargs) -> float:
        """Numerically evaluate with given variable assignments."""
        raise NotImplementedError

    def __float__(self):
        return self.eval()

    def latex(self) -> str:
        return str(self)


def _to_expr(x) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, float)):
        return Number(x)
    raise TypeError(f'Cannot convert {type(x)} to Expr')


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------

class Symbol(Expr):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(('Symbol', self.name))

    def subs(self, replacements: dict) -> Expr:
        return _to_expr(replacements[self.name]) if self.name in replacements else self

    def _diff(self, var: 'Symbol') -> Expr:
        return Number(1) if self.name == var.name else Number(0)

    def eval(self, **kwargs) -> float:
        return float(kwargs[self.name])

    def latex(self) -> str:
        return self.name


class Number(Expr):
    def __init__(self, value):
        self.value = float(value)

    def __str__(self):
        if self.value == int(self.value):
            return str(int(self.value))
        return str(self.value)

    def __hash__(self):
        return hash(('Number', self.value))

    def subs(self, replacements: dict) -> Expr:
        return self

    def _diff(self, var: 'Symbol') -> Expr:
        return Number(0)

    def eval(self, **kwargs) -> float:
        return self.value

    def latex(self) -> str:
        return str(self)


# Constants
pi = Number(math.pi)
E = Number(math.e)
I = Symbol('i')  # imaginary unit


# ---------------------------------------------------------------------------
# Binary operations
# ---------------------------------------------------------------------------

class BinOp(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def subs(self, replacements):
        return type(self)(self.left.subs(replacements), self.right.subs(replacements))


class Add(BinOp):
    def __str__(self):
        r = str(self.right)
        if r.startswith('-'):
            return f'({self.left} - {r[1:]})'
        return f'({self.left} + {self.right})'

    def _diff(self, var):
        return Add(self.left._diff(var), self.right._diff(var))

    def eval(self, **kw):
        return self.left.eval(**kw) + self.right.eval(**kw)


class Sub(BinOp):
    def __str__(self):
        return f'({self.left} - {self.right})'

    def _diff(self, var):
        return Sub(self.left._diff(var), self.right._diff(var))

    def eval(self, **kw):
        return self.left.eval(**kw) - self.right.eval(**kw)


class Mul(BinOp):
    def __str__(self):
        return f'({self.left}*{self.right})'

    def _diff(self, var):
        # Product rule
        return Add(Mul(self.left._diff(var), self.right),
                   Mul(self.left, self.right._diff(var)))

    def eval(self, **kw):
        return self.left.eval(**kw) * self.right.eval(**kw)


class Div(BinOp):
    def __str__(self):
        return f'({self.left}/{self.right})'

    def _diff(self, var):
        # Quotient rule
        return Div(Sub(Mul(self.left._diff(var), self.right),
                       Mul(self.left, self.right._diff(var))),
                   Pow(self.right, Number(2)))

    def eval(self, **kw):
        return self.left.eval(**kw) / self.right.eval(**kw)


class Pow(BinOp):
    def __str__(self):
        return f'({self.left}^{self.right})'

    def _diff(self, var):
        # Power rule (constant exponent)
        if isinstance(self.right, Number):
            n = self.right.value
            return Mul(Number(n), Mul(Pow(self.left, Number(n - 1)), self.left._diff(var)))
        # General: d/dx(f^g) = f^g * (g' ln(f) + g f'/f)
        return Mul(self, Add(
            Mul(self.right._diff(var), Func('log', self.left)),
            Div(Mul(self.right, self.left._diff(var)), self.left)
        ))

    def eval(self, **kw):
        return self.left.eval(**kw) ** self.right.eval(**kw)

    def latex(self) -> str:
        return f'{{{self.left.latex()}}}^{{{self.right.latex()}}}'


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

class Func(Expr):
    _funcs = {
        'sin': (math.sin, lambda arg: Func('cos', arg)),
        'cos': (math.cos, lambda arg: Mul(Number(-1), Func('sin', arg))),
        'tan': (math.tan, lambda arg: Pow(Func('cos', arg), Number(-2))),
        'exp': (math.exp, lambda arg: Func('exp', arg)),
        'log': (math.log, lambda arg: Div(Number(1), arg)),
        'sqrt': (math.sqrt, lambda arg: Div(Number(1), Mul(Number(2), Func('sqrt', arg)))),
        'abs': (abs, lambda arg: Func('sign', arg)),
    }

    def __init__(self, name: str, arg: Expr):
        self.name = name
        self.arg = arg

    def __str__(self):
        return f'{self.name}({self.arg})'

    def __hash__(self):
        return hash(('Func', self.name, str(self.arg)))

    def subs(self, replacements):
        return Func(self.name, self.arg.subs(replacements))

    def _diff(self, var):
        entry = self._funcs.get(self.name)
        if entry is None:
            return Number(0)
        _, d_fn = entry
        # Chain rule: d/dx f(g(x)) = f'(g(x)) * g'(x)
        return Mul(d_fn(self.arg), self.arg._diff(var))

    def eval(self, **kw):
        entry = self._funcs.get(self.name)
        if entry is None:
            raise ValueError(f'Unknown function: {self.name}')
        fn, _ = entry
        return fn(self.arg.eval(**kw))

    def latex(self) -> str:
        name_map = {'sin': '\\sin', 'cos': '\\cos', 'tan': '\\tan',
                    'exp': '\\exp', 'log': '\\log', 'sqrt': '\\sqrt'}
        n = name_map.get(self.name, self.name)
        return f'{n}({self.arg.latex()})'


def sin(x): return Func('sin', _to_expr(x))
def cos(x): return Func('cos', _to_expr(x))
def tan(x): return Func('tan', _to_expr(x))
def exp(x): return Func('exp', _to_expr(x))
def log(x): return Func('log', _to_expr(x))
def sqrt(x): return Func('sqrt', _to_expr(x))


# ---------------------------------------------------------------------------
# Equation
# ---------------------------------------------------------------------------

class Eq(Expr):
    def __init__(self, lhs: Expr, rhs: Expr):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f'{self.lhs} = {self.rhs}'

    def subs(self, replacements):
        return Eq(self.lhs.subs(replacements), self.rhs.subs(replacements))

    def _diff(self, var):
        return Eq(self.lhs._diff(var), self.rhs._diff(var))

    def eval(self, **kw):
        return abs(self.lhs.eval(**kw) - self.rhs.eval(**kw)) < 1e-10


# ---------------------------------------------------------------------------
# API functions
# ---------------------------------------------------------------------------

def symbols(names: str):
    """Create Symbol objects from a space- or comma-separated string."""
    parts = re.split(r'[\s,]+', names.strip())
    syms = [Symbol(p) for p in parts if p]
    return syms[0] if len(syms) == 1 else tuple(syms)


def diff(expr: Expr, var: Symbol, n: int = 1) -> Expr:
    """Differentiate expr n times with respect to var."""
    return expr.diff(var, n)


def simplify(expr: Expr) -> Expr:
    """Simplify expression (fold constants, remove zero/identity terms)."""
    if isinstance(expr, Number):
        return expr
    if isinstance(expr, Symbol):
        return expr
    if isinstance(expr, BinOp):
        l = simplify(expr.left)
        r = simplify(expr.right)
        if isinstance(expr, Add):
            if isinstance(l, Number) and isinstance(r, Number):
                return Number(l.value + r.value)
            if isinstance(r, Number) and r.value == 0:
                return l
            if isinstance(l, Number) and l.value == 0:
                return r
            return Add(l, r)
        elif isinstance(expr, Sub):
            if isinstance(l, Number) and isinstance(r, Number):
                return Number(l.value - r.value)
            if isinstance(r, Number) and r.value == 0:
                return l
            return Sub(l, r)
        elif isinstance(expr, Mul):
            if isinstance(l, Number) and isinstance(r, Number):
                return Number(l.value * r.value)
            if isinstance(l, Number) and l.value == 0:
                return Number(0)
            if isinstance(r, Number) and r.value == 0:
                return Number(0)
            if isinstance(l, Number) and l.value == 1:
                return r
            if isinstance(r, Number) and r.value == 1:
                return l
            return Mul(l, r)
        elif isinstance(expr, Div):
            if isinstance(l, Number) and isinstance(r, Number) and r.value != 0:
                return Number(l.value / r.value)
            return Div(l, r)
        elif isinstance(expr, Pow):
            if isinstance(r, Number) and r.value == 0:
                return Number(1)
            if isinstance(r, Number) and r.value == 1:
                return l
            if isinstance(l, Number) and isinstance(r, Number):
                return Number(l.value ** r.value)
            return Pow(l, r)
    if isinstance(expr, Func):
        return Func(expr.name, simplify(expr.arg))
    return expr


def expand(expr: Expr) -> Expr:
    """Expand products of sums (basic algebraic expansion)."""
    # Expand (a + b) * (c + d) → ac + ad + bc + bd
    if isinstance(expr, Mul):
        l = expand(expr.left)
        r = expand(expr.right)
        if isinstance(l, Add):
            return expand(Add(Mul(l.left, r), Mul(l.right, r)))
        if isinstance(r, Add):
            return expand(Add(Mul(l, r.left), Mul(l, r.right)))
        return Mul(l, r)
    if isinstance(expr, BinOp):
        return type(expr)(expand(expr.left), expand(expr.right))
    return expr


def subs(expr: Expr, **kwargs) -> Expr:
    """Substitute variables in expression."""
    return expr.subs(kwargs)


def integrate(expr: Expr, var: Symbol, a: Optional[float] = None,
              b: Optional[float] = None, n: int = 1000) -> Union[Expr, float]:
    """Numerical definite integral via Simpson's rule, or symbolic (basic)."""
    if a is not None and b is not None:
        # Numerical integration via Simpson's rule
        h = (b - a) / n
        xs = [a + i * h for i in range(n + 1)]
        ys = [expr.eval(**{var.name: float(x)}) for x in xs]
        s = ys[0] + ys[-1]
        s += 4 * sum(ys[i] for i in range(1, n, 2))
        s += 2 * sum(ys[i] for i in range(2, n - 1, 2))
        return float(h / 3 * s)
    # Basic symbolic integration (power rule + linearity)
    if isinstance(expr, Number):
        return Mul(expr, var)
    if isinstance(expr, Symbol):
        if expr.name == var.name:
            return Div(Pow(expr, Number(2)), Number(2))
        return Mul(expr, var)  # treat other symbols as constants
    if isinstance(expr, Pow) and isinstance(expr.right, Number):
        n_val = expr.right.value
        if n_val != -1:
            return Div(Pow(expr.left, Number(n_val + 1)), Number(n_val + 1))
        # 1/x → log|x|
        return Func('log', expr.left)
    # Linearity: ∫(f + g) = ∫f + ∫g
    if isinstance(expr, Add):
        return Add(integrate(expr.left, var), integrate(expr.right, var))
    if isinstance(expr, Sub):
        return Sub(integrate(expr.left, var), integrate(expr.right, var))
    # ∫(c * f) = c * ∫f when c is a constant
    if isinstance(expr, Mul):
        if isinstance(expr.left, Number):
            return Mul(expr.left, integrate(expr.right, var))
        if isinstance(expr.right, Number):
            return Mul(integrate(expr.left, var), expr.right)
    if isinstance(expr, Func) and expr.name == 'sin':
        return Mul(Number(-1), Func('cos', expr.arg))
    if isinstance(expr, Func) and expr.name == 'cos':
        return Func('sin', expr.arg)
    if isinstance(expr, Func) and expr.name == 'exp':
        return Func('exp', expr.arg)
    if isinstance(expr, Func) and expr.name == 'log':
        return Sub(Mul(expr.arg, Func('log', expr.arg)), expr.arg)
    if isinstance(expr, Func) and expr.name == 'tan':
        return Mul(Number(-1), Func('log', Func('cos', expr.arg)))
    if isinstance(expr, Func) and expr.name == 'sqrt':
        return Mul(Div(Number(2), Number(3)), Pow(expr.arg, Number(1.5)))
    if isinstance(expr, Func) and expr.name == 'sinh':
        return Func('cosh', expr.arg)
    if isinstance(expr, Func) and expr.name == 'cosh':
        return Func('sinh', expr.arg)
    if isinstance(expr, Func) and expr.name == 'tanh':
        return Func('log', Func('cosh', expr.arg))
    if isinstance(expr, Div):
        # ∫(f/g) — handle simple 1/f cases
        if isinstance(expr.left, Number) and expr.left.value == 1:
            # 1/x^n → x^(1-n)/(1-n) for n != 1; 1/x → log
            if isinstance(expr.right, Symbol) and expr.right.name == var.name:
                return Func('log', expr.right)
            if isinstance(expr.right, Pow) and isinstance(expr.right.right, Number):
                n_val = expr.right.right.value
                if n_val != 1:
                    new_exp = Number(1 - n_val)
                    return Div(Pow(expr.right.left, new_exp), new_exp)
        return Mul(expr, var)  # fallback: treat as constant * var
    # Last resort: treat expression as constant → c * var
    return Mul(expr, var)


def solve(eq: Eq, var: Symbol) -> Optional[Expr]:
    """Solve simple linear equations (ax + b = c form)."""
    # Rearrange to lhs - rhs = 0
    expr = Sub(eq.lhs, eq.rhs)
    # Try numerical Newton's method
    f = lambda x_val: expr.eval(**{var.name: x_val})
    # Try a range
    for x0 in [0.0, 1.0, -1.0, 2.0, -2.0, 10.0, -10.0]:
        try:
            x = float(x0)
            for _ in range(100):
                fx = f(x)
                eps = 1e-7
                dfx = (f(x + eps) - f(x - eps)) / (2 * eps)
                if abs(dfx) < 1e-14:
                    break
                x_new = x - fx / dfx
                if abs(x_new - x) < 1e-12:
                    return Number(round(x_new, 10))
                x = x_new
        except Exception:
            continue
    return None


__all__ = [
    'Expr', 'Symbol', 'Number', 'Add', 'Sub', 'Mul', 'Div', 'Pow',
    'Func', 'Eq', 'pi', 'E',
    'symbols', 'diff', 'simplify', 'expand', 'subs', 'integrate', 'solve',
    'sin', 'cos', 'tan', 'exp', 'log', 'sqrt',
]
