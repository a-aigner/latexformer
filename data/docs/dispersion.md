| Domain                             | Weight  |
| ---------------------------------- | ------- |
| Core algebra & arithmetic          | **22%** |
| Functions & calculus               | **18%** |
| Linear algebra (matrices, vectors) | **15%** |
| Sets, logic, relations             | **10%** |
| Series, sums, products             | **8%**  |
| Probability & statistics           | **7%**  |
| Complex numbers                    | **5%**  |
| Geometry & analytic geometry       | **5%**  |
| Discrete math & combinatorics      | **4%**  |
| Number theory                      | **3%**  |
| Vector calculus & operators        | **2%**  |
| Special functions & symbols        | **1%**  |
| Formatting stress / exotic LaTeX   | **1%**  |

# Domain-by-domain breakdown

## 1. Core Algebra & Arithmetic (22%)

### Why
This is the backbone of nearly all mathematical LaTeX. Also the largest source of user input.

### Unique LaTeX coverage
- Superscripts / subscripts  
- Fractions  
- Parentheses and nesting  
- Absolute values  
- Polynomials  
- Equations and inequalities  

### Examples
- `ax^2 + bx + c`  
- `\frac{a+b}{c-d}`  
- `|x - y|`  
- `x^n`  
- `a \le b`  

### Notes
This domain drives precedence, canonicalization, and paraphrase density.

---

## 2. Functions & Calculus (18%)

### Why
Heavy use of operators, limits, integrals, and spacing commands.

### Unique LaTeX coverage
- `\int`, `\lim`, `\frac{d}{dx}`, `\partial`  
- Superscripts/subscripts on operators  
- Differential spacing `\, dx`  
- Function application conventions  

### Examples
- `\int_0^1 x^2 \, dx`  
- `\lim_{x \to 0} \frac{\sin x}{x}`  
- `\frac{d}{dx} f(x)`  
- `\nabla f`  

### Notes
Important for consistency rules and operator formatting.

---

## 3. Linear Algebra (15%)

### Why
Introduces environments and alignment, which are structurally different from inline math.

### Unique LaTeX coverage
- `pmatrix`, `bmatrix`, `vmatrix`  
- Vector notation  
- Matrix operations  
- Systems of equations  

### Examples
- `\begin{pmatrix} a & b \\ c & d \end{pmatrix}`  
- `A\mathbf{x} = \mathbf{b}`  
- `\det(A)`  
- `\begin{cases} ... \end{cases}`  

### Notes
High impact: failures here are catastrophic.

---

## 4. Sets, Logic & Relations (10%)

### Why
Covers symbolic logic, set builder notation, and relational operators.

### Unique LaTeX coverage
- `\{ \mid \}`  
- `\forall`, `\exists`  
- `\in`, `\subset`, `\cap`, `\cup`  
- Logical connectives  

### Examples
- `\{ x \in \mathbb{R} \mid x > 0 \}`  
- `A \cap B`  
- `x \notin S`  
- `\forall x \exists y`  

---

## 5. Series, Sums & Products (8%)

### Why
Tests operator limits, stacked notation, and grouping.

### Unique LaTeX coverage
- `\sum`, `\prod`  
- Multi-line subscripts/superscripts  
- Infinite bounds  

### Examples
- `\sum_{i=1}^n i^2`  
- `\prod_{k=1}^\infty (1 + \frac{1}{k})`  

---

## 6. Probability & Statistics (7%)

### Why
Introduces bracket notation and semantic operators.

### Unique LaTeX coverage
- `P(A \mid B)`  
- `E[X]`  
- `\mathrm{Var}(X)`  
- Distributions  

### Examples
- `P(A \cap B)`  
- `E[X^2]`  
- `\sigma^2`  

---

## 7. Complex Numbers (5%)

### Why
Covers imaginary unit, Euler notation, polar forms.

### Unique LaTeX coverage
- `i`  
- `\Re`, `\Im`  
- `e^{i\theta}`  
- Modulus and argument  

### Examples
- `z = a + bi`  
- `|z|`  
- `e^{i\pi} + 1 = 0`  

---

## 8. Geometry & Analytic Geometry (5%)

### Why
Adds coordinate-based formulas and implicit equations.

### Unique LaTeX coverage
- Coordinate systems  
- Conic sections  
- Distance formulas  

### Examples
- `x^2 + y^2 = r^2`  
- `\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}`  

---

## 9. Discrete Math & Combinatorics (4%)

### Why
Adds combinatorial notation and recurrence structures.

### Unique LaTeX coverage
- `\binom{n}{k}`  
- Recurrence relations  
- Graph-theoretic symbols  

### Examples
- `\binom{n}{k}`  
- `a_n = a_{n-1} + a_{n-2}`  

---

## 10. Number Theory (3%)

### Why
Adds specialized operators and functions.

### Unique LaTeX coverage
- `\gcd`, `\lcm`  
- Modular arithmetic  
- Divisibility  

### Examples
- `\gcd(a,b)`  
- `a \equiv b \pmod{n}`  

---

## 11. Vector Calculus & Operators (2%)

### Why
Specialized but introduces operator notation and bold symbols.

### Unique LaTeX coverage
- `\nabla`  
- `\cdot`, `\times`  
- Vector fields  

### Examples
- `\nabla \cdot \mathbf{F}`  
- `\nabla \times \mathbf{F}`  

---

## 12. Special Functions & Symbols (1%)

### Why
Long-tail coverage and completeness.

### Unique LaTeX coverage
- Greek letters  
- Bessel, Gamma, Zeta  
- Special operators  

### Examples
- `\Gamma(x)`  
- `\zeta(s)`  
- `J_\nu(x)`  

---

## 13. Formatting Stress & Exotic LaTeX (1%)

### Why
Stress-test LaTeX grammar, not math semantics.

### Unique LaTeX coverage
- `\left...\right`  
- Nested environments  
- Overset/underset  
- Spacing commands  

### Examples
- `\left( \frac{a}{b} \right)`  
- `\overset{*}{x}`  
