"""nimblend knows dimensions, labels, entries and alignment, and nothing else.

The dependency runs one way, and the tests that hold it live in the consumer:
they scan this package from outside. These hold the other half from inside,
which is what a package developed on its own needs — its own suite fails when
the vocabulary of the layer above arrives, whether or not the consumer's
suite is run.
"""

import ast
from pathlib import Path

import nimblend as nb

# words that belong to the layer above and mean nothing about a labeled
# array. `row` and `column` are not among them: an index matrix has rows and
# columns, and CSR names a row pointer and column indices, so they are
# scanned in names alone, below.
FOREIGN = (
    "constraint",
    "objective",
    "solver",
    "variable",
    "dual",
    "primal",
    "optimi",
    "feasib",
    "nimopt",
)

# a name is the tighter rule: prose may say an index matrix is read by row,
# and no function, parameter or class here may be named for one.
FOREIGN_NAMES = FOREIGN + ("row", "column")


def modules():
    """Every module of the package, as a parsed tree beside its path."""
    root = Path(nb.__file__).parent
    return [(path, ast.parse(path.read_text())) for path in sorted(root.rglob("*.py"))]


def declared_names(tree):
    """Every name this module declares: classes, functions and parameters."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            found.append((node.lineno, node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append((node.lineno, node.name))
            args = node.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs:
                found.append((node.lineno, arg.arg))
            for arg in (args.vararg, args.kwarg):
                if arg is not None:
                    found.append((node.lineno, arg.arg))
    return found


def test_no_name_in_the_package_belongs_to_the_layer_above():
    offenders = [
        f"{path.name}:{line} {name} <- {word}"
        for path, tree in modules()
        for line, name in declared_names(tree)
        for word in FOREIGN_NAMES
        if word in name.lower()
    ]
    assert offenders == [], offenders


def test_no_source_of_the_package_states_a_word_of_the_layer_above():
    # docstrings included: a docstring explaining what a constraint wants is
    # the vocabulary arriving, one release before the function does
    offenders = [
        f"{path.name} <- {word}"
        for path, _ in modules()
        for word in FOREIGN
        if word in path.read_text().lower()
    ]
    assert offenders == [], offenders


def test_the_package_declares_no_dependency_but_numpy():
    # the structural half: an import of the layer above fails here, because
    # nothing installs it beside this package
    root = Path(nb.__file__).parents[2]
    text = (root / "pyproject.toml").read_text()
    declared = text.split("dependencies = ", 1)[1].split("]", 1)[0]
    assert "numpy" in declared
    assert "nimopt" not in declared


def test_the_scan_catches_a_name_of_the_layer_above():
    caught = [
        name
        for _, name in declared_names(ast.parse("def write_row(constraint): pass\n"))
        for word in FOREIGN_NAMES
        if word in name.lower()
    ]
    assert sorted(caught) == ["constraint", "write_row"]
