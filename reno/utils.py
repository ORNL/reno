"""Utility functions that are potentially needed in multiple modules."""

import importlib.resources
import inspect
from typing import Any

import numpy as np

import reno

# these are the default colors to use when rendering 'selected'
# references in latex. Might want to change in disgusting lightmode contexts
DOC_HIGHLIGHT_COLOR = "teal"
EQ_HIGHLIGHT_COLOR = "cyan"


def _get_assigned_var_name(var: Any) -> str:
    """In a model context manager we don't have the direct ability to get the name of the variable
    we're assigning to the model (to avoid the PyMC requirement of specifying the name separately
    as a string.)

    To address that, this function pulls from the local frame to find the variable matching the
    one passed.

    https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
    """
    # NOTE: this can't be used within the __init__ of a reference because the
    # variable on the left hand side of the assignment isn't technically in the
    # frame yet/hasn't been created.
    frame_items = inspect.currentframe().f_back.f_back.f_locals.items()
    for var_name, var_val in frame_items:
        if var_val is var:
            return var_name

    # TODO: should this throw an error instead?
    return None


def find_ref_root_model(ref):
    """Find the top most parent model in the submodel tree."""
    model = ref.model
    while model.parent is not None:
        model = model.parent
    return model


def ref_universe(
    refs: list["reno.components.TrackedReference"], depth=1
) -> list["reno.components.TrackedReference"]:
    """Fancier recursive seek_refs, useful for collecting the full set of
    references to and from any in the given set of passed refs.

    Given a set of references, expand outwards ``depth`` times to find other
    references used.

    Args:
        refs (list[reno.components.TrackedReference]): The references to start expanding outwards from.
        depth (int): How many steps outward to take finding dependencies from the initial references list.

    Returns:
        A list (including the initial references) of expanded dependencies/dependents.
    """
    universe = [*refs]
    while depth > 0:
        new_refs = []
        for ref in universe:
            # check for "inputs"
            if isinstance(ref, reno.components.Stock):
                new_refs.extend(ref._implied_eq().seek_refs())
            else:
                new_refs.extend(ref.seek_refs())

            # check for "outputs"
            for ref2 in find_ref_root_model(ref).all_refs():
                if isinstance(ref2, reno.components.Stock):
                    if ref in ref2._implied_eq().seek_refs():
                        new_refs.append(ref2)
                else:
                    if ref in ref2.seek_refs():
                        new_refs.append(ref2)

        universe.extend(new_refs)
        universe = list(set(universe))
        depth -= 1

    return universe


def is_ref_in_parent_scope(ref, source) -> bool:
    """Check if a reference is in the scope _above_ some source reference.
    (In other words, in the source model's parent, or parent of that parent,
    up the chain, anything except the same model as source)

    This is primarily used in the diagramming logic to determine which
    model/submodel needs to handle rendering a particular reference.

    Args:
        ref (reno.components.TrackedReference): The reference to look for in
            parent scopes.
        source (reno.components.TrackedReference): The reference to search
            the parent chain through.

    Returns:
        ``True`` if a ref is somewhere in the source's parent chain
        (and notably _not_ in the same exact model)
    """
    if ref.model is None or source.model is None:
        # TODO: arguably if source's model is NOT None and ref's is,
        # that should count as rendering "external to the current model"
        return False

    parent = source.model.parent
    while parent is not None:
        if ref.model == parent:
            return True
        parent = parent.parent
    return False


def ensure_scalar(operand: Any) -> Any:
    """Convert int/float types into a Scalar type if relevant, otherwise
    directly return what was passed."""
    if isinstance(operand, (int, float, np.ndarray)):
        return reno.components.Scalar(operand)
    if isinstance(operand, list):
        return reno.components.Scalar(np.array(operand))
    return operand


def is_free_var(eq: "reno.components.EquationPart") -> bool:
    """A reference is a "free reference/free variable" if its equation
    has no references to _other_ references.

    Expects an EquationPart rather than only the ref itself because this applies
    both to stock .init as well as flow/var .eq
    """
    if (
        isinstance(
            eq,
            (
                reno.components.Scalar,
                reno.components.Distribution,
                int,
                float,
                np.ndarray,
            ),
        )
        or eq is None
    ):
        return True
    return False


def check_for_easy_static_time_eq(eq: "reno.components.EquationPart") -> bool:
    """Determine whether an equation is `t - STATIC_EQ`.

    This is necessary for basic pymc optimizations on accessing historical values.
    (Basically any index equations not following this form are gonna be a whole lot
    harder, and I'm not going to worry about implementing them yet.)
    """

    if not isinstance(eq, reno.ops.sub):
        return False
    if not isinstance(eq.sub_equation_parts[0], reno.components.TimeRef):
        return False
    if (
        not is_static(eq.sub_equation_parts[1])
        or len(eq.sub_equation_parts[1].find_refs_of_type(reno.components.Distribution))
        > 0
    ):
        # NOTE: currently not counting distributions as "easy" because of
        # weirdness in computing the index multiple times resulting in multiple
        # values (thus pymc taps are wrong)
        return False
    return True


def is_static(
    eq: "reno.components.EquationPart",
    checked: list["reno.components.Reference"] = None,
) -> bool:
    """An equation is "static" if it only contains values that won't change, e.g. scalars,
    distributions, or other static variables. Specifically stock and time references are
    what make an equation _not_ static.

    Args:
        eq (EquationPart): The equation to determine static status of.
        checked (list[Reference]): Used internally to avoid infinite recursion in checking,
            don't manually pass.
    """
    # print("Called is_static on", eq)
    if eq is None:
        # really?? I guess this makes sense, none isn't going to change if it even runs.
        # print("No eq, so not static")
        return True
    if checked is None:
        checked = []
    # the reason we're throwing in ref checks instead of just checking
    # isinstance type is because you could theoretically have an eq of
    # e.g. Scalar(4) + Scalar(5) which is obviously still static
    for ref in eq.seek_refs():
        # print("Checking", ref)
        # the hasattr is to check for already non-tracked references (e.g. TimeRef)
        if not hasattr(ref, "_static"):
            # print("No _static, so not static")
            return False

        # TODO: TODO: TODO: WHAT A MESS. CLEAN THIS UP.

        # if ref.eq is not None:
        if ref not in checked or ref._static is None:
            checked.append(ref)
            if not is_static(ref, checked):
                return False
        else:
            # TODO: we don't set _static in here though, this would only
            # happen from TrackedReference._determine_if_static. (which is
            # similarly calling this is_static function)
            # Is the expectation that this condition is only for when that
            # sequence of _determine_if_static() calls during population?
            # It feels really weird to be using a piece of state for what
            # should likely be a stateless check?
            if not ref._static:
                # print("Ref's _static is false, so not static")
                return False

        # if ref.eq is not None and ref not in checked:
        #     # something something without checked we got inf recursion
        #     # sometimes??
        #     checked.append(ref)
        #     if not is_static(ref.eq, checked):
        #         return False
        # else:
        #     if not ref.static:  # TODO: and ref.static_computed? (no, that's something else)
        #         return False

        # if not ref.static or (ref.eq is not None and ref not in checked and not is_static(ref.eq, checked+ref)):
        #     return False
        #     # already determined to be static
        #     continue
        # if ref.eq is not None and not is_static(ref.eq):
        #     return False
    if isinstance(eq, reno.components.Distribution) and eq.per_timestep:
        # print("Is distribution with per_timestep, not static")
        return False
    if isinstance(
        eq,
        (reno.components.Scalar, int, float, np.ndarray),
    ):
        # keeping this explicitly to make it obvious
        # print("Equation is a scalar, int, float, or array, is static!")
        return True
    if isinstance(
        eq,
        (
            reno.components.Stock,
            reno.components.TimeRef,
            reno.components.HistoricalValue,
        ),
    ):
        # anything dealing with a stock (inherently updated every timestep) or
        # is time itself, is obv not static
        # print("Explicit time-based thing included, not static")
        return False
    if isinstance(eq, reno.ops.slice):
        # slices can potentially index time-based things (if stop is None or explicitly a
        # non-static variable) which would make this non-static
        if not eq.sub_equation_parts[0].is_static():
            return False
        if (
            eq.stop is None
            or (eq.stop is not None and not eq.stop.is_static())
            or (eq.start is not None and not eq.start.is_static())
        ):
            # print("Slice with non-static start/stop, not static")
            return False

    if isinstance(eq, reno.ops.orient_timeseries):
        return False
    if len(eq.find_refs_of_type(reno.ops.orient_timeseries)) > 0:
        return False
    # if hasattr(eq, "eq" and isinstance(eq, reno.ops.orient_timeseries):
    #     return False

    # UGH. I'm just playing whackamole (wow that word looks like guacamole) with
    # weird edgecases I keep accidentally adding.
    if (
        hasattr(eq, "eq")
        and isinstance(eq.eq, reno.components.Distribution)
        and eq.eq.per_timestep
    ):
        # print("Sub equation is a per_timestep distribution, not static")
        return False
    # this was really only created for one specific use-case, a technically
    # non-static distribution (with per_timestep specified) doesn't trigger any
    # of the above when you're calling is_static on _the containing reference_.
    # if isinstance(eq, (reno.components.Variable, reno.components.Flow)):
    #     return is_static(eq.eq, checked)
    # print("Nothing else triggered, is static")
    return True


def latex_name(name: str, cmd: str = "text") -> str:
    """Wrap a reference name in a latex string, correctly escaping anything as necessary.
    It will _not_ escape if there's a '$', indicating 'yes use mathmode' like in latex.
    This allows reference labels like `"$x_3$"`

    Example:
        >>> latex_name("testing")
        "\\text{testing}"

        >>> latex_name("x_3", "texttt")
        "\\texttt{x\\_3}"
    """
    if "$" in name:
        return f"\\{cmd}{{{name}}}"
    escaped_name = name.replace("_", "\\_")
    return f"\\{cmd}{{{escaped_name}}}"


def latex_eqline_wrap_doc(doc_text: str, highlight: bool = False) -> str:
    """Wrap a docstring in an appropriate latex block for an equation set display.

    (This handles adding a highlight color if it's a selected reference.)
    """
    string = "& \\mkern36mu "
    if highlight:
        string += "{\\color{" + DOC_HIGHLIGHT_COLOR + "} \\text{" + doc_text + "}}"
    else:
        string += "{\\color{grey} \\text{" + doc_text + "}}"
    string += " \\\\\n"
    return string


def latex_eqline_wrap(eq_text: str, highlight: bool = False) -> str:
    """Wrap an equation in an appropriate latex block for an equation set display.

    (This handles adding a highlight color if it's a selected reference.)
    """
    string = "& "
    if highlight:
        string += "{\\color{" + EQ_HIGHLIGHT_COLOR + "} " + eq_text + "}"
    else:
        string += eq_text
    string += " \\\\\n"
    return string


def range_eq_latex(min_eq, max_eq, **kwargs) -> str:
    """Get the latex string for displaying the combined min/max range of a reference."""
    if min_eq is None and max_eq is None:
        return ""

    range_eq = "\\mkern18mu \\in"

    if min_eq is None:
        min_text = "(-\\infty,"
    else:
        min_text = "[" + min_eq.latex(**kwargs) + ","

    if max_eq is None:
        max_text = "\\infty)"
    else:
        max_text = max_eq.latex(**kwargs) + "]"

    return range_eq + min_text + max_text


def get_dependency_relevant_equation_part(
    init_eqs: bool, ref: "reno.components.Reference"
) -> "reno.components.EquationPart":
    """Determine what equation part should be used to run seek_refs on
    in order to determine compute dependencies."""
    if init_eqs and isinstance(ref, reno.components.TrackedReference):
        if ref.init is None:
            # if inits were requested but no explicit init equation found
            if isinstance(ref, reno.components.Stock):
                # The init for a stock with no init just defaults to 0.0
                return reno.components.Scalar(0.0)
            # for vars and flows just use the regular equation if no init
            # specified
            return ref._implied_eq()
        return ref._implied_eq(ref.init)
    elif init_eqs and not isinstance(ref, reno.components.TrackedReference):
        # metrics would reach this I think
        # NOTE: technically this is unnecessary/handled by else below, leaving
        # it to make it explicit which condition we expect metrics to fall under
        return ref
    else:
        # NOTE: because of seek_refs behavior, for tracked references,
        # min/max ("implied eq") already gets taken care of, so returning
        # ref works in all cases.
        return ref


def dependency_compute_order(
    refs: list["reno.components.Reference"], init_eqs: bool = False, debug: bool = False
) -> list["reno.components.Reference"]:
    """Find a dependency-safe ordering for reference equations by iterating through
    and each time adding the first reference that doesn't depend on any references
    not yet added.

    This function will detect circular references (equations that depend on eachother)
    and throw an error. Primary use for this function is to correctly set order of
    equations in the pymc model/step function.

    Args:
        refs (list[reno.components.Reference]): The references to find a safe compute
            order for.
        init_eqs (bool): Use init equations to determine ordering, where available.
        debug (bool): Spit out everything it's analyzing.

    Returns:
        The dependency-ordered list of reno references passed in.
    """

    compute_order = []
    remaining = []
    iterations = 0

    # start with every ref in remaining
    remaining.extend(refs)

    # I think technically iterations should never exceed the total number of
    # references (every iteration should find one new thing that can be
    # computed), but doubling just in case
    while len(remaining) > 0 and iterations < len(refs) * 2:
        if debug:
            print(compute_order, remaining)
        ref_to_remove = None
        for obj in remaining:
            if debug:
                print("\t", obj.qual_name())
            missing_dependency = False

            eq_part = get_dependency_relevant_equation_part(init_eqs, obj)

            all_dependencies = eq_part.seek_refs()
            # disregard anything we haven't been explicitly asked about, and
            # don't include self (should only occur for seek refs of a stock
            # implied eq)
            potential_dependencies = [
                d for d in all_dependencies if d in refs and d != obj
            ]

            if debug:
                print("\t\tAll dependencies:", all_dependencies)
                print("\t\tRelevant dependencies:", potential_dependencies)
            for ref in potential_dependencies:
                if isinstance(ref, reno.components.TimeRef):
                    # time refs don't count because they have no additional
                    # dependencies and are always provided at compute time
                    continue
                if isinstance(ref, reno.components.HistoricalValue):
                    # the assumption for now is that computing a historical
                    # value's time index is static (minus timeref) and since the
                    # result was already computed in the past - no resulting
                    # refs are considered a dependency
                    continue

                if debug:
                    print("\t\t", ref)
                if ref not in compute_order:
                    if debug:
                        try:
                            print("\t\t\tCan't compute yet because of", ref.qual_name())
                        except:  # noqa: E722
                            print("???")
                    # we can't compute this obj yet, a necessary ref won't have
                    # been computed beforehand
                    missing_dependency = True
                    break

            if not missing_dependency:
                # if we made it to this point, no uncomputed refs, so we're good
                # to compute this one next
                if debug:
                    print("\t\tWe're good!")
                compute_order.append(obj)
                ref_to_remove = obj
                break
        if ref_to_remove is not None:
            remaining.remove(ref_to_remove)
        iterations += 1

    if len(remaining) > 0:
        raise Exception(
            f"Got stuck in dependency loop, could not determine compute order for {remaining}"
        )

    return compute_order


def resource_path(filename: str) -> str:
    """Get the path to the package "data resource" with matching filename. This is
    intended to get supporting js/vue files etc.

    Args:
        filename (str): the name of the resource file to get the full path of.

    Returns:
        The full package resource file path for specified filename.
    """
    path = None
    with importlib.resources.as_file(
        importlib.resources.files("reno") / "res" / filename
    ) as full_file_path:
        path = str(full_file_path)
    return path
