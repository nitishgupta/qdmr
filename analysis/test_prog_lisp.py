from typing import List

from parse_dataset.parse import (qdmr_program_to_nested_expression, nested_expression_to_lisp,
                                 remove_args_from_nested_expression)



def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == "(":
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(")", ""))
        while token[-1] == ")":
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]


def print_sexp(exp):
    out = ''
    if type(exp) == type([]):
        out += '(' + ' '.join(print_sexp(x) for x in exp) + ')'
    elif type(exp) == type('') and re.search(r'[\s()]', exp):
        out += '"%s"' % repr(exp)[1:-1].replace('"', '\"')
    else:
        out += '%s' % exp
    return out


def annotation_to_lisp_exp(annotation: str) -> str:
    # TODO: Remove this hard-coded fix
    annotation = annotation.replace('and\n', 'bool_and\n')
    annotation = annotation.replace('or\n', 'bool_or\n')

    expressions = annotation.split('\n')
    output_depth = 0
    output = []

    def count_depth(exp: str):
        """count the depth of this expression. Every dot in the prefix symbols a depth entry."""
        return len(exp) - len(exp.lstrip('.'))

    def strip_attention(exp: str):
        """remove the [attention] part of the expression"""
        if '[' in exp:
            return exp[:exp.index('[')]
        else:
            return exp

    for i, exp in enumerate(expressions):
        depth = count_depth(exp)
        if i + 1 < len(expressions):
            next_expression_depth = count_depth(expressions[i+1])
        else:
            next_expression_depth = 0

        output.append('(')

        exp = strip_attention(exp)
        exp = exp.lstrip('.')
        output.append(exp)

        if next_expression_depth <= depth:
            # current clause should be closed
            output.append(')')

        while next_expression_depth < depth:
            # close until currently opened depth
            output.append(')')
            depth -= 1

        output_depth = depth

    while 0 < output_depth:
        output.append(')')
        output_depth -= 1

    # now make sure there's no one-expression in a parentheses (e.g. "(exist (find))" which should be "(exist find)")
    i = 0
    new_output = []
    while i < len(output):
        exp = output[i]
        if i + 2 >= len(output):
            new_output.append(exp)
            i += 1
            continue

        exp1 = output[i+1]
        exp2 = output[i+2]

        if exp == '(' and exp1 not in ['(', ')'] and exp2 == ')':
            new_output.append(exp1)
            i += 2
        else:
            new_output.append(exp)

        i += 1

    output = ' '.join(new_output)
    output = output.replace('( ', '(')
    output = output.replace(' )', ')')
    return output

def annotation_to_module_attention(annotation: str) -> List:
    """
    retrieves the raw annotation string and extracts the word indices attention for each module
    """
    lines = annotation.split('\n')
    attn_supervision = []
    for line in lines:
        # We assume valid input, that is, each line either has no brackets at all,
        # or has '[' before ']', where there are numbers separated by commas between.
        if '[' in line:
            start_i = line.index('[')
            end_i = line.index(']')
            module = line[:start_i].strip('.')
            sentence_indices = line[start_i+1:end_i].split(',')
            sentence_indices = [ind.strip() for ind in sentence_indices]

            attn_supervision.append((module, sentence_indices))
    return attn_supervision

# input_program = \
# "equal\n"\
# ".count\n"\
# "..with_relation[3]\n"\
# "...find[1]\n"\
# "...project[5,6]\n"\
# "....find[8,9,10]\n"\
# ".1[0]\n"
#
# lisp_program = annotation_to_lisp_exp(input_program)
# python_program = lisp_to_nested_expression(lisp_program)
# attn_supervision = annotation_to_module_attention(input_program)
# back_lisp = nested_expression_to_lisp(python_program)
# print(back_lisp)
# print(lisp_program)
# print(python_program)
# print(back_lisp)
# print((back_lisp == lisp_program))
# print(attn_supervision)

print("\n")

x = "(COMPARISON (max (SELECT North__China West__India) (SELECT North__Middle)))"
nested_python_program = lisp_to_nested_expression(x)
back_lisp = nested_expression_to_lisp(nested_python_program)
print(nested_python_program)
print(back_lisp)
print()

# program =  [
#             "SELECT['when was Jabal Shammar being annexed']",
#             "SELECT['when was the preliminary attack on Taif']",
#             "COMPARISON['max', '#1', '#2']"
#         ]

program = [
            "SELECT['people killed According to official reports']",
            "AGGREGATE['count', '#1']",
            "SELECT['people wounded According to official reports']",
            "AGGREGATE['count', '#3']",
            "COMPARISON['max', '#2', '#4']"
        ]
qdmr_nested_expression = qdmr_program_to_nested_expression(program)
print(qdmr_nested_expression)
print(nested_expression_to_lisp(qdmr_nested_expression))
print(lisp_to_nested_expression(nested_expression_to_lisp(qdmr_nested_expression)))

print(remove_args_from_nested_expression(qdmr_nested_expression))