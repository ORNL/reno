"""Functions to handle parsing strings of reno ops/equations into the actual
reno op objects themselves. This is primarily needed for the interative viz
portions and the ability to serialize models for saving/loading to file."""

import json

import reno


def parse(
    string: str, refs: dict[str, "reno.components.Reference"] = None
) -> reno.components.EquationPart:
    """Turn an equation string into Reno's math equation objects. This runs prefix parsing on
    equation strings/reprs, e.g. '(+ (- 5 'my_variable') 3)' to turn it into a fully populated
    EquationPart.

    This function runs recursively through each level of an equation tree.

    Args:
        string (str): The string representation (repr) of a Reno equation tree: "(+ (- 5 'my_variable'))"
        refs (dict[str, reno.components.Reference]): Any references that have already been populated, so
            multiple ``'my_variable'`` strings translate to the same object correctly.

    Returns:
        An EquationPart populated with recursive sub_equation_parts.
    """
    string = string.strip()

    if refs is None:
        refs = {}

    table = parser_table()

    # check if we need to do python style parsing (e.g. a distribution definition)
    try:
        class_or_scalar_conversion = parse_class_or_scalar(string)
        if isinstance(class_or_scalar_conversion, (float, int)):
            return reno.components.Scalar(class_or_scalar_conversion)
        return class_or_scalar_conversion
    except SyntaxError:
        # no handling needed, this just means we couldn't parse a python
        # constructor call out of the string, continue with normal prefix
        # parsing
        pass
    # TODO: missing handling of bool/int etc? Why am I not doing
    # parse_value here?
    # try_simple_convert_first = parse_value(string)
    # if isinstance(try_simple_convert_first, (float, int)):
    #     print("Found float/int")
    #     return reno.components.Scalar(try_simple_convert_first)

    # check for string (likely a reference?)
    if (string.startswith('"') and string.endswith('"')) or (
        string.startswith("'") and string.endswith("'")
    ):
        if string[1:-1] in refs:
            return refs[string[1:-1]]
        return string[1:-1]
        # TODO: not clear if this should actually error or throw a warning or
        # what. Probably at least a warning is warranted.
        # if string[1:-1] not in refs:
        #     raise SyntaxError(f"Reference {string} not found/undefined")

    op_name, arg_strs = parse_op_str(string)
    if op_name not in table:
        raise SyntaxError(f"Invalid operation or reference '{op_name}'")

    # pull out the corresponding python class for this operation
    op_class = table[op_name]

    # if a class has a specific way it needs to parse, use that (e.g. piecewise
    # and history)
    if hasattr(op_class, "parse"):
        return op_class.parse(arg_strs, refs)

    # otherwise do a normal recursive parse of any arguments
    parsed_args = []
    for arg_str in arg_strs:
        parsed_args.append(parse(arg_str, refs))

    # initialize the actual operation object (EquationPart)
    return op_class(*parsed_args)


def parse_value(string: str) -> float | int | str:
    """Try to parse out a float or int if possible, otherwise just return the string itself.

    e.g. '5.0', or '13'
    """
    val = None

    # try to convert to int
    if val is None:
        try:
            val = int(string)
        except ValueError:
            val = None

    # try to convert to float
    if val is None:
        try:
            val = float(string)
        except ValueError:
            val = None

    # leaving out for now, not sure where this
    # would be needed?
    # try to convert to bool
    # if val is None:
    #     try:
    #         val = bool(string)
    #     except ValueError:
    #         val = None

    if val is None:
        if str(string).startswith("[") and str(string).endswith("]"):
            # safer parse of a list than a eval() would be
            val = json.loads(string)

    # leave as string if it's none of the above types
    if val is None:
        val = string
    return val


def parse_op_str(string: str, no_op: bool = False) -> tuple[str, list[str]]:
    """Pull out the **topmost** operation and component arguments, leaving
    all nested/sub components as raw strings. This effectively pulls out just
    the data needed to construct the next root of an equation tree.

    Args:
        string (str): The string of serialized operations/equation repr.
        no_op (bool): Specify ``True`` if you have a string that only contains
            arguments, useful for custom parse functions in certain classes
            (e.g. piecewise)

    Returns:
        A tuple with the operation name string, and a list of strings for
        each argument found (again only at the top level.)
    """
    # remove outermost parens
    string = string.strip()
    if string.startswith("(") and string.endswith(")"):
        string = string[1:-1]

    if not no_op:
        # get operation name
        if " " not in string:
            raise SyntaxError(f"No op name found in '{string}', missing space?")

        op_name = string[: string.index(" ")]
        string = string[string.index(" ") + 1 :]

    arg_strs = []

    parens_stack = 0
    # we track how many parentheses deep because this function only handles the
    # "top" level
    arg_start_i = 0
    for i in range(len(string)):
        if string[i] == "(":
            parens_stack += 1
        elif string[i] == ")":
            parens_stack -= 1
        elif string[i] == " " and parens_stack == 0:
            arg_strs.append(string[arg_start_i:i])
            arg_start_i = i + 1

        if i == len(string) - 1:
            if parens_stack != 0:
                raise SyntaxError(f"Unmatched '(' in '{string}'")
            arg_strs.append(string[arg_start_i:])

    if no_op:
        return arg_strs
    return op_name, arg_strs


def parser_table() -> dict[str, type]:
    """Get a dictionary of operation names and their associated python types/classes."""
    table = {}
    for op_class in reno.components.Operation.op_types():
        table[op_class.op_repr()] = op_class

    # add special cases like history and piecewise
    table["piecewise"] = reno.components.Piecewise
    table["history"] = reno.components.HistoricalValue

    return table


# ^^^ -- prefix parsing -- ^^^
# ----------------------------------------------------------------------
# vvv -- python func syntax parsing -- vvv


def parse_function_args(string: str) -> tuple[list[any], dict[str, any], int, int]:
    """Pull out any python formatted args or kwargs for a function.

    e.g. 'Normal(5.0, std=1.0)'
    """
    if "(" not in string:
        raise SyntaxError(f"Was attempting to find '(' for parameters of '{string}'")
    if ")" not in string:
        raise SyntaxError(f"Was attempting to find ')' for parameters of '{string}'")
    start = string.index("(") + 1
    end = string.index(")")

    pieces = []
    braces_stack = 0
    last_start = start
    for i in range(start, end):
        if string[i] == "[":
            braces_stack += 1
        elif string[i] == "]":
            braces_stack -= 1
        elif string[i] == "," and braces_stack == 0:
            pieces.append(string[last_start:i])
            last_start = i + 1

        if i == end - 1:
            if braces_stack != 0:
                raise SyntaxError(f"Unmatched '['] in '{string}'")
            pieces.append(string[last_start:end])

    args = []
    kwargs = {}
    # pieces = string[start:end].split(",")  # doesn't account for array args
    for piece in pieces:
        # check if arg or kwarg
        if "=" in piece:
            key = piece[: piece.index("=")].strip()
            value = piece[piece.index("=") + 1 :].strip()
            kwargs[key] = parse_value(value)
            # TODO: missing parsing of lists, bools, etc.
        else:
            args.append(parse_value(piece))

    return args, kwargs, start, end


def parse_class_or_scalar(string) -> reno.components.EquationPart:
    """Parse a single non-math op concatenated equation part, e.g.
    a scalar (float or int) or distribution with parameters."""
    string = string.strip()
    if string == "" or string == "None":
        return None

    # check if it's just a float or int
    try_simple_convert_first = parse_value(string)
    if isinstance(try_simple_convert_first, (float, int)):
        return try_simple_convert_first

    # must be an op, pull the params
    args, kwargs, start, end = parse_function_args(string)
    op_name = string[: start - 1].strip()

    classes = [
        reno.ops.Normal,
        reno.ops.Uniform,
        reno.ops.DiscreteUniform,
        reno.ops.Bernoulli,
        reno.ops.Categorical,
        reno.components.Scalar,
    ]

    for c in classes:
        if c.__name__ == op_name:
            return c(*args, **kwargs)

    raise SyntaxError(f"Couldn't parse '{string}'")
