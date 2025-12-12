# Paraphrase Families (15%)
### Purpose
Teach the model that many linguistic descriptions map to the same mathematical structure, without changing meaning or form. This is the core generalization mechanism.
### Design Rules
- Output LaTeX must be identical across the family.
- Do not introduce algebraic manipulation.
- Vary phrasing, syntax, word order, and verbosity.
- Include both concise and verbose formulations.
### Addresses
- Overfitting to fixed phrases
- Brittleness to real user language
- Lexical variation issues
#### Examples
Input: x squared plus y squared
Output: x^2 + y^2

Input: the sum of x to the power of two and y to the power of two
Output: x^2 + y^2

Input: add the square of x and the square of y
Output: x^2 + y^2

Input: x raised to two added to y raised to two
Output: x^2 + y^2

# Notation Variants & Synonym Mappings (15%)
### Purpose
Normalize natural-language synonyms into a single LaTeX representation.
### Design Rules
- Decide one canonical LaTeX form and enforce it.
- Cover prepositions (“over”, “per”, “divided by”).
- Include function naming variations (“log base 10”, “common log”).
### Addresses
- Inconsistent outputs
- Fragmented token learning
- Real-world phrasing diversity
#### Examples
Input: a divided by b
Output: \frac{a}{b}

Input: the ratio of a to b
Output: \frac{a}{b}

Input: a over b
Output: \frac{a}{b}

Input: the quotient when a is divided by b
Output: \frac{a}{b}

# Operator Precedence & Grouping Emphasis (15%)
### Purpose
Teach structural disambiguation, especially parentheses and binding order.
### Design Rules
- Include minimal-parentheses and explicit-grouping variants.
- Use language cues like “the quantity”, “the sum of”, “the product of”.
- Never rely on implicit algebraic rewriting.
### Addresses
- Parentheses errors
- Precedence hallucinations
- Structural ambiguity
#### Examples
Input: a plus b times c
Output: a + bc

Input: a plus the product of b and c
Output: a + bc

Input: the quantity a plus b times c
Output: (a + b)c

Input: the product of a plus b and c
Output: (a + b)c

# Object Structures (15%)
Matrices, Systems, Piecewise, Sets etc.
### Purpose
Ensure correct LaTeX environments and structure, not just symbols.
### Design Rules
- Always use a consistent environment (pmatrix, cases, etc.).
- Keep descriptions explicit about dimensions and entries.
- Do not perform row operations or simplifications.
### Addresses
- Broken environments
- Misplaced alignment tokens
- Structural LaTeX errors
#### Examples
Input: a two by two matrix with entries a b c d
Output: \begin{pmatrix} a & b \ c & d \end{pmatrix}

Input: a system where x plus y equals one and x minus y equals zero
Output: \begin{cases} x + y = 1 \ x - y = 0 \end{cases}

Input: a piecewise function equal to x squared for x nonnegative and minus x otherwise
Output: \begin{cases} x^2 & x \ge 0 \ -x & x < 0 \end{cases}

# LaTeX Syntax & Style Consistency (10%)
### Purpose
Force uniform formatting choices so the model outputs one style only.
### Design Rules
- Fix all conventions:
    - \sin(x) not \sin x
    - Always \frac{} not a/b
    - Explicit parentheses
- Repetition is intentional.
### Addresses
- Style drift
- Inconsistent outputs
- Post-processing complexity
#### Examples
Input: sine of x
Output: \sin(x)

Input: sin x
Output: \sin(x)

Input: the sine function applied to x
Output: \sin(x)

# Normalization & Canonicalization (10%)
### Purpose
Reduce equivalent representations to a single canonical LaTeX form.
### Design Rules
Only normalize explicitly allowed properties (e.g., commutativity).
Never simplify numerically.
Keep transformations reversible in principle.
### Addresses
Output entropy
Equivalent-form explosion
Training instability
#### Examples
Input: y plus x
Output: x + y

Input: x plus y
Output: x + y

Input: b times a
Output: ab

Input: a times b
Output: ab

# Error Correction (5%)
LaTeX & Conceptual (4% Latex + 1% Conceptual)
### Purpose
Teach the model to repair malformed input, not to “solve” problems.
### Design Rules
- Errors must be minimal and local.
- Output should be the corrected expression only.
- Conceptual fixes must be universally true identities.
### Addresses
- User typos
- Model robustness
- Downstream parse failures
#### Examples
##### LaTeX syntax
Input: \frac{x+1{2}
Output: \frac{x+1}{2}

Input: \sin x)
Output: \sin(x)

##### Conceptual (limited)
Input: sin(x)^2 + cos(x)^2 equals 2
Output: \sin^2(x) + \cos^2(x) = 1