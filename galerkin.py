"""
1D spectral Galerkin framework for assembling mass/stiffness-like operators
and projecting/solving ODEs and PDEs with different polynomial and
trigonometric bases and their boundary-condition lifts.

`FunctionSpace(N, domain)` is the base class; `Legendre`/`Chebyshev` are
orthogonal polynomial spaces (reference (-1, 1)), `Sines`/`Cosines` are
trigonometric spaces (reference (0, 1)), and `Composite` spaces such as
`DirichletChebyshev` enforce boundary conditions via a stencil matrix `S`
and a lift `B`. `TrialFunction`/`TestFunction`/`inner` form a small
weak-form DSL used together with `project` and `assemble_generic_matrix`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal, Self, cast, overload

import numpy as np
import sympy as sp
from numpy.polynomial import Chebyshev as Cheb
from numpy.polynomial import Legendre as Leg
from numpy.polynomial._polybase import ABCPolyBase
from scipy import sparse
from scipy.integrate import quad

x: sp.Symbol = sp.Symbol("x")

Domain = tuple[float, float]
BC = tuple[float | sp.Expr, float | sp.Expr]
BasisCallable = Callable[[np.ndarray], np.ndarray]
Matrix = np.ndarray | sparse.spmatrix


@overload
def map_reference_domain(x: np.ndarray, d: Domain, r: Domain) -> np.ndarray: ...
@overload
def map_reference_domain(x: sp.Expr, d: Domain, r: Domain) -> sp.Expr: ...
def map_reference_domain(x, d, r):
    """Map point(s) x from the true domain d to the reference domain r."""
    return r[0] + (r[1] - r[0]) * (x - d[0]) / (d[1] - d[0])


@overload
def map_true_domain(x: np.ndarray, d: Domain, r: Domain) -> np.ndarray: ...
@overload
def map_true_domain(x: sp.Expr, d: Domain, r: Domain) -> sp.Expr: ...
def map_true_domain(x, d, r):
    """Map point(s) x from the reference domain r to the true domain d."""
    return d[0] + (d[1] - d[0]) * (x - r[0]) / (r[1] - r[0])


def map_expression_true_domain(u: sp.Basic, x: sp.Symbol, d: Domain, r: Domain) -> sp.Basic:
    """Rewrite the sympy expression u(x), given on domain d, in terms of the reference domain r."""
    if d != r:
        u = sp.sympify(u)
        xm = map_true_domain(x, d, r)
        u = u.replace(x, xm)
    return u


class FunctionSpace(ABC):
    """Abstract base class for a 1D spectral function space of N+1 basis
    functions on `domain`.

    Concrete subclasses provide `basis_function`/`derivative_basis_function`
    (and typically a faster `eval`/`mass_matrix`); this base class implements
    the shared machinery: mesh generation, numerical inner products, and a
    generic (quadrature-based, slower) mass matrix.
    """

    def __init__(self, N: int, domain: Domain = (-1, 1)) -> None:
        self.N = N
        self._domain = domain

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def domain_factor(self) -> float:
        d = self.domain
        r = self.reference_domain
        return (d[1] - d[0]) / (r[1] - r[0])

    def mesh(self, N: int | None = None) -> np.ndarray:
        d = self.domain
        n = N if N is not None else self.N
        return np.linspace(d[0], d[1], n + 1)

    def weight(self, x: sp.Symbol = x) -> sp.Expr | int:
        return 1

    @abstractmethod
    def L2_norm_sq(self, N: int) -> np.ndarray: ...

    @property
    @abstractmethod
    def reference_domain(self) -> Domain: ...

    @overload
    def basis_function(self, j: int, sympy: Literal[True]) -> sp.Expr: ...
    @overload
    def basis_function(self, j: int, sympy: Literal[False] = False) -> BasisCallable: ...
    @abstractmethod
    def basis_function(self, j, sympy=False): ...

    @abstractmethod
    def derivative_basis_function(self, j: int, k: int = 1) -> BasisCallable: ...

    def evaluate_basis_function(self, Xj: np.ndarray, j: int) -> np.ndarray:
        return self.basis_function(j)(Xj)

    def evaluate_derivative_basis_function(
        self, Xj: np.ndarray, j: int, k: int = 1
    ) -> np.ndarray:
        return self.derivative_basis_function(j, k=k)(Xj)

    def eval(self, uh: np.ndarray, xj: np.ndarray | float) -> np.ndarray:
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh

    def eval_basis_function_all(self, Xj: np.ndarray) -> np.ndarray:
        P = np.zeros((len(Xj), self.N + 1))
        for j in range(self.N + 1):
            P[:, j] = self.evaluate_basis_function(Xj, j)
        return P

    def inner_product(self, u: sp.Expr) -> np.ndarray:
        us = map_expression_true_domain(u, x, self.domain, self.reference_domain)
        us = sp.lambdify(x, us)
        uj = np.zeros(self.N + 1)
        h = self.domain_factor
        r = self.reference_domain
        for i in range(self.N + 1):
            psi = self.basis_function(i)

            def uv(Xj, psi=psi):
                return us(Xj) * psi(Xj)

            uj[i] = float(h) * quad(uv, float(r[0]), float(r[1]))[0]
        return uj

    def mass_matrix(self) -> Matrix:
        """Generic (slow) mass matrix via numerical quadrature; subclasses may override with a closed-form version."""
        return assemble_generic_matrix(TrialFunction(self), TestFunction(self))


class Polynomial(FunctionSpace):
    """Base class for polynomial function spaces on reference domain (-1, 1)."""

    @property
    def reference_domain(self) -> Domain:
        return (-1, 1)

    @overload
    def basis_function(self, j: int, sympy: Literal[True]) -> sp.Expr: ...
    @overload
    def basis_function(self, j: int, sympy: Literal[False] = False) -> ABCPolyBase: ...
    @abstractmethod
    def basis_function(self, j, sympy=False): ...

    def derivative_basis_function(self, j: int, k: int = 1) -> ABCPolyBase:
        return self.basis_function(j).deriv(k)

    def mass_matrix(self) -> sparse.spmatrix:
        m = self.L2_norm_sq(self.N + 1)
        return sparse.diags(m, offsets=0, format="csr")


class Legendre(Polynomial):
    """Orthogonal Legendre polynomial space, P_0..P_N, on reference domain (-1, 1)."""

    def __init__(self, N: int, domain: Domain = (-1, 1)) -> None:
        FunctionSpace.__init__(self, N, domain=domain)

    @overload
    def basis_function(self, j: int, sympy: Literal[True]) -> sp.Expr: ...
    @overload
    def basis_function(self, j: int, sympy: Literal[False] = False) -> Leg: ...
    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.legendre(j, x)
        return Leg.basis(j)

    def L2_norm_sq(self, N: int) -> np.ndarray:
        # raise NotImplementedError
        return 2 / (2 * np.arange(N) + 1)

    def eval(self, uh: np.ndarray, xj: np.ndarray | float) -> np.ndarray:
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        return np.polynomial.legendre.legval(Xj, uh)


class Chebyshev(Polynomial):
    """Orthogonal Chebyshev polynomial space, T_0..T_N, on reference domain (-1, 1)."""

    def __init__(self, N: int, domain: Domain = (-1, 1)) -> None:
        FunctionSpace.__init__(self, N, domain=domain)

    @overload
    def basis_function(self, j: int | sp.Symbol, sympy: Literal[True]) -> sp.Expr: ...
    @overload
    def basis_function(self, j: int, sympy: Literal[False] = False) -> Cheb: ...
    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.cos(j * sp.acos(x))
        return Cheb.basis(j)

    def weight(self, x: sp.Symbol = x) -> sp.Expr:
        return 1 / sp.sqrt(1 - x**2)

    def L2_norm_sq(self, N: int) -> np.ndarray:
        # raise NotImplementedError
        return 2 ** (np.arange(N) == 0) * np.pi / 2

    def eval(self, uh: np.ndarray, xj: np.ndarray | float) -> np.ndarray:
        # raise NotImplementedError
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        return np.polynomial.chebyshev.chebval(Xj, uh)

    def inner_product(self, u: sp.Expr) -> np.ndarray:
        """Inner products (u, T_i), via the substitution x = cos(theta), integrating over
        [0, pi] to handle the 1/sqrt(1-x^2) weight."""
        us = map_expression_true_domain(u, x, self.domain, self.reference_domain)
        # change of variables to x=cos(theta)
        us = sp.simplify(us.subs(x, sp.cos(x)), inverse=True)
        us = sp.lambdify(x, us)
        uj = np.zeros(self.N + 1)
        h = float(self.domain_factor)
        k = sp.Symbol("k")
        basis = sp.lambdify(
            (k, x),
            sp.simplify(self.basis_function(k, True).subs(x, sp.cos(x), inverse=True)),
        )
        for i in range(self.N + 1):

            def uv(Xj, j):
                return us(Xj) * basis(j, Xj)

            uj[i] = float(h) * quad(uv, 0, np.pi, args=(i,))[0]
        return uj


class Trigonometric(FunctionSpace):
    """Base class for trigonometric function spaces on reference domain (0, 1).

    Adds a boundary lift in `eval` via `self.B.Xl(X)`; concrete subclasses
    (`Sines`, `Cosines`) provide the basis functions and set `self.B`.
    """

    B: Dirichlet | Neumann

    @property
    def reference_domain(self) -> Domain:
        return (0, 1)

    def mass_matrix(self) -> sparse.spmatrix:
        m = self.L2_norm_sq(self.N + 1)
        return sparse.diags([m], [0], (self.N + 1, self.N + 1), format="csr")

    def eval(self, uh: np.ndarray, xj: np.ndarray | float) -> np.ndarray:
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh + self.B.Xl(Xj)


class Sines(Trigonometric):
    """Sine basis sin((j+1)*pi*x); homogeneous Dirichlet plus a `Dirichlet` lift `B`."""

    def __init__(self, N: int, domain: Domain = (0, 1), bc: BC = (0, 0)) -> None:
        Trigonometric.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)

    @overload
    def basis_function(self, j: int, sympy: Literal[True]) -> sp.Expr: ...
    @overload
    def basis_function(self, j: int, sympy: Literal[False] = False) -> BasisCallable: ...
    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.sin((j + 1) * sp.pi * x)
        return lambda Xj: np.sin((j + 1) * np.pi * Xj)

    def derivative_basis_function(self, j: int, k: int = 1) -> BasisCallable:
        scale = ((j + 1) * np.pi) ** k * {0: 1, 1: -1}[(k // 2) % 2]
        if k % 2 == 0:
            return lambda Xj: scale * np.sin((j + 1) * np.pi * Xj)
        else:
            return lambda Xj: scale * np.cos((j + 1) * np.pi * Xj)

    def L2_norm_sq(self, N: int) -> np.ndarray:
        return np.full(N, 0.5)


class Cosines(Trigonometric):
    def __init__(
        self, N: int, domain: Domain = (0, 1), bc: BC = (0, 0), constraint: int = 0
    ) -> None:
        # raise NotImplementedError
        Trigonometric.__init__(self, N, domain=domain)
        self.B = Neumann(bc, domain, self.reference_domain)

    @overload
    def basis_function(self, j: int, sympy: Literal[True]) -> sp.Expr: ...
    @overload
    def basis_function(self, j: int, sympy: Literal[False] = False) -> BasisCallable: ...
    def basis_function(self, j, sympy=False):
        # raise NotImplementedError
        if sympy:
            return sp.cos((j) * sp.pi * x)
        return lambda Xj: np.cos((j) * np.pi * Xj)

    def derivative_basis_function(self, j: int, k: int = 1) -> BasisCallable:
        # raise NotImplementedError
        scale = ((j) * np.pi) ** k * {0: 1, 1: -1}[(k // 2) % 2]
        if k % 2 == 0:
            return lambda Xj: scale * np.cos((j) * np.pi * Xj)
        else:
            return lambda Xj: scale * -np.sin((j) * np.pi * Xj)

    def L2_norm_sq(self, N: int) -> np.ndarray:
        # raise NotImplementedError
        res = np.full(N, 0.5)
        res[0] = 1.0
        return res


# Create classes to hold the boundary function


class Dirichlet:
    """Affine lift `B.x` on `domain` with `B.x(d0) = bc[0]`, `B.x(d1) = bc[1]`;
    `B.Xl` evaluates it on `reference_domain`."""

    def __init__(self, bc: BC, domain: Domain, reference_domain: Domain) -> None:
        d = domain
        r = reference_domain
        h = d[1] - d[0]
        self.bc = bc
        self.x = (
            bc[0] * (d[1] - x) / h + bc[1] * (x - d[0]) / h
        )  # in physical coordinates
        self.xX = map_expression_true_domain(
            self.x, x, d, r
        )  # in reference coordinates
        self.Xl = sp.lambdify(x, self.xX)


class Neumann:
    """Quadratic lift `B.x` on `domain` with `B.x'(d0) = bc[0]`, `B.x'(d1) = bc[1]`;
    `B.Xl` evaluates it on `reference_domain`."""

    def __init__(self, bc: BC, domain: Domain, reference_domain: Domain) -> None:
        d = domain
        r = reference_domain
        h = d[1] - d[0]
        self.bc = bc
        self.x = bc[0] / h * (d[1] * x - x**2 / 2) + bc[1] / h * (
            x**2 / 2 - d[0] * x
        )  # in physical coordinates
        self.xX = map_expression_true_domain(
            self.x, x, d, r
        )  # in reference coordinates
        self.Xl = sp.lambdify(x, self.xX)


class Composite(FunctionSpace):
    r"""Base class for function spaces created as linear combinations of orthogonal basis functions

    The composite basis functions are defined using the orthogonal basis functions
    (Chebyshev or Legendre) and a stencil matrix S. The stencil matrix S is used
    such that basis function i is

    .. math::

        \psi_i = \sum_{j=0}^N S_{ij} Q_j

    where :math:`Q_i` can be either the i'th Chebyshev or Legendre polynomial

    For example, both Chebyshev and Legendre have Dirichlet basis functions

    .. math::

        \psi_i = Q_i-Q_{i+2}

    Here the stencil matrix will be

    .. math::

        s_{ij} = \delta_{ij} - \delta_{i+2, j}, \quad (i, j) \in (0, 1, \ldots, N) \times (0, 1, \ldots, N+2)

    Note that the stencil matrix is of shape :math:`(N+1) \times (N+3)`.
    """

    B: Dirichlet | Neumann
    S: sparse.csr_matrix
    L2_norm_sq: Callable[[int], np.ndarray]

    def eval(self, uh: np.ndarray, xj: np.ndarray | float) -> np.ndarray:
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh + self.B.Xl(Xj)

    def mass_matrix(self) -> sparse.spmatrix:
        M = sparse.diags(
            [self.L2_norm_sq(self.N + 3)],
            [0],
            shape=(self.N + 3, self.N + 3),
            format="csr",
        )
        return self.S @ M @ self.S.T


class DirichletLegendre(Composite, Legendre):
    def __init__(self, N: int, domain: Domain = (-1, 1), bc: BC = (0, 0)) -> None:
        Legendre.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)
        self.S = sparse.diags((1., -1.), (0, 2), shape=(N + 1, N + 3), format="csr")

    @overload
    def basis_function(self, j: int, sympy: Literal[True]) -> sp.Expr: ...
    @overload
    def basis_function(self, j: int, sympy: Literal[False] = False) -> Leg: ...
    def basis_function(self, j, sympy=False):
        # raise NotImplementedError
        if sympy:
            return sp.legendre(j, x) - sp.legendre(j + 2, x)
        return Leg.basis(j) - Leg.basis(j + 2)


class NeumannLegendre(Composite, Legendre):
    def __init__(
        self, N: int, domain: Domain = (-1, 1), bc: BC = (0, 0), constraint: int = 0
    ) -> None:
        # raise NotImplementedError
        Legendre.__init__(self, N, domain=domain)
        self.B = Neumann(bc, domain, self.reference_domain)
        j = np.arange(N + 1)
        a = j * (j + 1) / ((j + 2) * (j + 3))
        self.S = sparse.diags((1., -a), (0, 2), shape=(N + 1, N + 3), format="csr")
        # self.S = sparse.diags((1., -1.), (0, 2), shape=(N + 1, N + 3), format="csr")

    @overload
    def basis_function(self, j: int, sympy: Literal[True]) -> sp.Expr: ...
    @overload
    def basis_function(self, j: int, sympy: Literal[False] = False) -> Leg: ...
    def basis_function(self, j, sympy=False):
        # raise NotImplementedError
        a = j * (j + 1) / ((j + 2) * (j + 3))
        if sympy:
            return sp.legendre(j, x) - a * sp.legendre(j + 2, x)
        return Leg.basis(j) - a * Leg.basis(j + 2)


class DirichletChebyshev(Composite, Chebyshev):
    def __init__(self, N: int, domain: Domain = (-1, 1), bc: BC = (0, 0)) -> None:
        Chebyshev.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)
        self.S = sparse.diags((1., -1.), (0, 2), shape=(N + 1, N + 3), format="csr")

    @overload
    def basis_function(self, j: int | sp.Symbol, sympy: Literal[True]) -> sp.Expr: ...
    @overload
    def basis_function(self, j: int, sympy: Literal[False] = False) -> Cheb: ...
    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.cos(j * sp.acos(x)) - sp.cos((j + 2) * sp.acos(x))
        return Cheb.basis(j) - Cheb.basis(j + 2)


class NeumannChebyshev(Composite, Chebyshev):
    def __init__(
        self, N: int, domain: Domain = (-1, 1), bc: BC = (0, 0), constraint: int = 0
    ) -> None:
        # raise NotImplementedError
        Chebyshev.__init__(self, N, domain=domain)
        self.B = Neumann(bc, domain, self.reference_domain)
        j = np.arange(N + 1)
        a = j ** 2 / ((j + 2) ** 2)
        self.S = sparse.diags((1., -a), (0, 2), shape=(N + 1, N + 3), format="csr")

    @overload
    def basis_function(self, j: int | sp.Symbol, sympy: Literal[True]) -> sp.Expr: ...
    @overload
    def basis_function(self, j: int, sympy: Literal[False] = False) -> Cheb: ...
    def basis_function(self, j, sympy=False):
        a = j ** 2 / ((j + 2) ** 2)
        # raise NotImplementedError
        if sympy:
            return sp.cos(j * sp.acos(x)) - a * sp.cos((j + 2) * sp.acos(x))
        return Cheb.basis(j) - a * Cheb.basis(j + 2)


class BasisFunction:
    """Base for `TrialFunction`/`TestFunction`: a symbolic function in `V`, `diff`
    derivatives deep, used as an argument to `inner`."""

    def __init__(self, V: FunctionSpace, diff: int = 0) -> None:
        self._V = V
        self._num_derivatives = diff

    @property
    def function_space(self) -> FunctionSpace:
        return self._V

    @property
    def num_derivatives(self) -> int:
        return self._num_derivatives

    def diff(self, k: int) -> Self:
        return self.__class__(self.function_space, diff=self.num_derivatives + k)


class TestFunction(BasisFunction):
    """Test function argument to `inner`, e.g. `v = TestFunction(V)`."""


class TrialFunction(BasisFunction):
    """Trial function argument to `inner`, e.g. `u = TrialFunction(V)`; supports `u.diff(k)`."""


def assemble_generic_matrix(u: TrialFunction, v: TestFunction) -> np.ndarray:
    """Assemble the bilinear form matrix (u^(du), v^(dv)) by numerical quadrature,
    using the correct weight (e.g. for Chebyshev) and exploiting symmetry when du == dv."""
    assert isinstance(u, TrialFunction)
    assert isinstance(v, TestFunction)
    V = v.function_space
    assert u.function_space == V
    r = V.reference_domain
    D = np.zeros((V.N + 1, V.N + 1))
    cheb = V.weight() == 1 / sp.sqrt(1 - x**2)
    symmetric = u.num_derivatives == v.num_derivatives
    w = {"weight": "alg" if cheb else None, "wvar": (-0.5, -0.5) if cheb else None}

    def uv(Xj, i, j):
        return V.evaluate_derivative_basis_function(
            Xj, i, k=v.num_derivatives
        ) * V.evaluate_derivative_basis_function(Xj, j, k=u.num_derivatives)

    for i in range(V.N + 1):
        for j in range(i if symmetric else 0, V.N + 1):
            D[i, j] = quad(uv, float(r[0]), float(r[1]), args=(i, j), **w)[0]
            if symmetric:
                D[j, i] = D[i, j]
    return D


@overload
def inner(u: sp.Expr, v: TestFunction) -> np.ndarray: ...
@overload
def inner(u: TrialFunction, v: TestFunction) -> Matrix: ...
def inner(u, v):
    """Weak-form inner product: (u, v) for a sympy expression u, or the bilinear
    form matrix (u, v) for a `TrialFunction` u, e.g. `inner(u.diff(2), v)`."""
    V = v.function_space
    h = V.domain_factor
    if isinstance(u, TrialFunction):
        num_derivatives = u.num_derivatives + v.num_derivatives
        if num_derivatives == 0:
            return float(h) * V.mass_matrix()
        else:
            return float(h) ** (1 - num_derivatives) * assemble_generic_matrix(u, v)
    return V.inner_product(u)


def project(ue: sp.Expr, V: FunctionSpace) -> np.ndarray:
    """L2-project the sympy expression ue onto the function space V; returns its expansion coefficients."""
    u = TrialFunction(V)
    v = TestFunction(V)
    b = inner(ue, v)
    A = inner(u, v)
    uh = sparse.linalg.spsolve(A, b)
    return uh


def L2_error(uh: np.ndarray, ue: sp.Expr, V: FunctionSpace, kind: str = "norm") -> float:
    """L2 norm of the error between the numerical solution V.eval(uh, .) and the exact expression ue."""
    d = V.domain
    uej = sp.lambdify(x, ue)

    def uv(xj):
        return ((uej(xj) - V.eval(uh, xj)) ** 2).item()

    return np.sqrt(quad(uv, float(d[0]), float(d[1]))[0])


def test_project() -> None:
    ue = sp.besselj(0, x)
    domain = (0, 10)
    for space in (Chebyshev, Legendre):
        V = space(16, domain=domain)
        u = project(ue, V)
        err = L2_error(u, ue, V)
        print(f"test_project: L2 error = {err:2.4e}, N = {V.N}, {V.__class__.__name__}")
        assert err < 1e-6


def test_helmholtz() -> None:
    ue = sp.besselj(0, x)
    f = ue.diff(x, 2) + ue
    domain = (0, 10)
    for space in (
        NeumannChebyshev,
        NeumannLegendre,
        DirichletChebyshev,
        DirichletLegendre,
        Sines,
        Cosines,
    ):
        if space in (NeumannChebyshev, NeumannLegendre, Cosines):
            bc = ue.diff(x, 1).subs(x, domain[0]), ue.diff(x, 1).subs(x, domain[1])
        else:
            bc = ue.subs(x, domain[0]), ue.subs(x, domain[1])
        N = 60 if space in (Sines, Cosines) else 12
        V = space(N, domain=domain, bc=bc)
        u = TrialFunction(V)
        v = TestFunction(V)
        A = cast(np.ndarray, inner(u.diff(2), v)) + cast(np.ndarray, inner(u, v))
        b = inner(f - (V.B.x.diff(x, 2) + V.B.x), v)
        u_tilde = np.linalg.solve(A, b)
        err = L2_error(u_tilde, ue, V)
        print(f"test_helmholtz: L2 error = {err:2.4e}, N = {N}, {V.__class__.__name__}")
        assert err < 1e-3, f"Failed for {V.__class__.__name__} with N={N}, error={err}"


def test_convection_diffusion() -> None:
    eps = 0.05
    ue = (sp.exp(-x / eps) - 1) / (sp.exp(-1 / eps) - 1)
    f = 0
    domain = (0, 1)
    for space in (DirichletLegendre, DirichletChebyshev, Sines):
        N = 50 if space is Sines else 16
        V = space(N, domain=domain, bc=(0, 1))
        u = TrialFunction(V)
        v = TestFunction(V)
        A = cast(np.ndarray, inner(u.diff(2), v)) + (1 / eps) * cast(
            np.ndarray, inner(u.diff(1), v)
        )
        b = inner(f - ((1 / eps) * V.B.x.diff(x, 1)), v)
        u_tilde = np.linalg.solve(A, b)
        err = L2_error(u_tilde, ue, V)
        print(
            f"test_convection_diffusion: L2 error = {err:2.4e}, N = {N}, {V.__class__.__name__}"
        )
        assert err < 1e-3


if __name__ == "__main__":
    test_project()
    test_convection_diffusion()
    test_helmholtz()
