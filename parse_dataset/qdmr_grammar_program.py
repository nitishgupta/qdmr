from typing import List
import json
import argparse

from qdmr.domain_languages.qdmr_language import QDMRLanguage
from qdmr.data.utils import Node, QDMRExample, nested_expression_to_lisp, nested_expression_to_tree, \
    string_arg_to_quesspan_pred, read_qdmr_json_to_examples

"""
*initial goal of this script; update after we actually implement this*
This script takes QDMR programs parsed using parse_dataset.parse_qdmr and uses the grammar proposed in QDMRLanguage
to see what programs can be produced from this grammar. This would require certain transformations due to quirks of the
QDMR annotation. Also, not every program can be successfully mapped to a typed-grammar. 

We're currently able to parse 7296/7519 program from DROP train
"""


qdmr_langugage = QDMRLanguage()
template2questionprogram = {}
template2count = {}
total_not_parsed = 0


def is_project_select_node(node: Node):
    # ['PROJECT', 'people of #REF', ['SELECT', 'nationalities registered in Bilbao']],
    # node.children = ['people of #REF', ['SELECT', 'nationalities registered in Bilbao']]
    # node.children[1] = ['SELECT', 'nationalities registered in Bilbao']

    if node.predicate == "PROJECT" and len(node.children) == 2 and len(node.children[1].children) == 1 and \
            node.children[1].predicate == "SELECT":
        return True
    else:
        return False


def create_partial_group_node(count_or_sum: str, project_string_arg: str):
    """Returns a node:
        ['PARTIAL_GROUP_count/sum',
            ['PARTIAL_PROJECT/PARTIAL_SELECT_NUM', GET_QUESTION_SPAN('project_string_arg')]
        ]

    """
    if count_or_sum == "count":
        partial_group_node = Node(predicate="PARTIAL_GROUP_count")
        partial_select_node = Node(predicate="PARTIAL_PROJECT")
    elif count_or_sum == "sum":
        partial_group_node = Node(predicate="PARTIAL_GROUP_sum")
        partial_select_node = Node(predicate="PARTIAL_SELECT_NUM")
    else:
        raise NotImplementedError
    partial_select_string_arg_node = Node(predicate="GET_QUESTION_SPAN", string_arg=project_string_arg)
    partial_select_node.add_child(partial_select_string_arg_node)
    partial_group_node.add_child(partial_select_node)
    return partial_group_node


def transform_to_fit_grammar(node: Node, question: str, top_level=False):
    """This function takes a program as a Tree, and converts predicates so the resulting program is type-conforming.

    QDMR, for example, could have a program such as, COMPARISON_max(SELECT, SELECT) which should ideally be
    COMPARISON_max(SELECT_NUM, SELECT_NUM)

    As a first step, we simply gather a list of predicates that require NUMBER arguments, and if their actual children
    are SELECT, we convert them to SELECT-NUM
    """

    skip_recursive_type_conforming = False

    # Trivia -- 330 programs in DROP_train were fixed by this!!
    if node.predicate in ["COMPARISON_max", "COMPARISON_min"]:
        # First check if all children are AGGREGATE_count(SELECT) or AGGREGATE_sum(SELECT)
        all_chilren_count_select = True
        all_chilren_sum_select = True
        for child in node.children:
            child_is_count_select = False
            child_is_sum_select = False
            if child.predicate in ["AGGREGATE_count"]:
                if len(child.children) == 1 and child.children[0].predicate == "SELECT":
                    child_is_count_select = True
            elif child.predicate in ["AGGREGATE_sum"]:
                if len(child.children) == 1 and child.children[0].predicate == "SELECT":
                    child_is_sum_select = True
            all_chilren_count_select = all_chilren_count_select and child_is_count_select
            all_chilren_sum_select = all_chilren_sum_select and child_is_sum_select

        if all_chilren_count_select or all_chilren_sum_select:
            # Need to remove AGGREGATE_count/AGGREGATE_sum from all children and change the function at this node.
            new_children = []
            for child in node.children:
                # From the test above: The child is AGGREGATE_count/sum and it only has one child that is SELECT
                new_child = child.children[0]
                new_children.append(new_child)
            # Replace old children with new ones
            node.children = []
            for c in new_children:
                node.add_child(c)
            predicate = ""
            if all_chilren_count_select:
                predicate = "COMPARISON_count_max" if node.predicate == "COMPARISON_max" else "COMPARISON_count_min"
            if all_chilren_sum_select:
                predicate = "COMPARISON_sum_max" if node.predicate == "COMPARISON_max" else "COMPARISON_sum_min"
            node.predicate = predicate

    elif node.predicate in ["COMPARATIVE"]:
        # CASE 1 --
        # ['COMPARATIVE',
        #   ['SELECT', 'nationalities registered in Bilbao'],
        #   ['GROUP_count',
        #       ['PROJECT', 'people of #REF', ['SELECT', 'nationalities registered in Bilbao']],
        #       ['SELECT', 'nationalities registered in Bilbao']],
        #   'is higher than 10']
        # ]
        # -->
        # ['COMPARATIVE',
        #   ['SELECT', 'nationalities registered in Bilbao'],
        #   ['PARTIAL_GROUP_count', ['PARTIAL_PROJECT', 'people of #REF']],
        #   ['CONDITION', 'is higher than 10']
        # ]
        # In case of GROUP_sum, PARTIAL_GROUP_count --> PARTIAL_GROUP_sum and PARTIAL_PROJECT --> PARTIAL_SELECT_NUM
        if (len(node.children) == 3 and
                node.children[0].predicate == "SELECT" and
                node.children[1].predicate in ["GROUP_count", "GROUP_sum"] and
                len(node.children[1].children) == 2 and
                node.children[1].children[0].predicate == "PROJECT" and
                node.children[1].children[1].predicate == "SELECT" and
                node.children[2].predicate == "GET_QUESTION_SPAN"):
            # We're in CASE 1
            select_string_arg = node.children[0].children[0].string_arg

            # Creating SELECT node
            select_node = Node(predicate="SELECT")
            select_arg_node = Node(predicate="GET_QUESTION_SPAN", string_arg=select_string_arg)
            select_node.add_child(select_arg_node)
            # GROUP_count vs. GROUP_sum
            project_string_arg = node.children[1].children[0].children[0].string_arg
            if node.children[1].predicate == "GROUP_count":
                # Creating ['PARTIAL_GROUP_count', ['PARTIAL_PROJECT', GET_QUESTION_SPAN('project_string_arg')]]
                partial_node = create_partial_group_node(count_or_sum="count", project_string_arg=project_string_arg)
            elif node.children[1].predicate == "GROUP_sum":
                # Creating ['PARTIAL_GROUP_sum', ['PARTIAL_SELECT_NUM', GET_QUESTION_SPAN('project_string_arg')]]
                partial_node = create_partial_group_node(count_or_sum="sum", project_string_arg=project_string_arg)
                pass
            else:
                raise NotImplementedError
            # Creating Condition node
            condition_string_arg = node.children[2].string_arg
            condition_node = Node(predicate="CONDITION")
            condition_string_arg_node = Node(predicate="GET_QUESTION_SPAN", string_arg=condition_string_arg)
            condition_node.add_child(condition_string_arg_node)

            # Replacing COMPARATIVE children to new ones made above
            new_children = [select_node, partial_node, condition_node]
            node.children = []
            for c in new_children:
                node.add_child(c)
        # CASE 2 --
        # ['COMPARATIVE',
        #   ['PROJECT', 'who threw #REF', ['SELECT', 'touchdown passes']],
        #   ['GROUP_count',
        #       ['SELECT', 'touchdown passes'],
        #       ['PROJECT', 'who threw #REF', ['SELECT', 'touchdown passes']]],
        #   'is the most']
        # ]
        # -->
        # ['COMPARATIVE',
        #   ['SELECT', 'who threw #REF'],
        #   ['PARTIAL_GROUP_count', ['PARTIAL_PROJECT', 'touchdown passes']],
        #   ['CONDITION', 'is the most']
        # ]
        elif (len(node.children) == 3 and
                is_project_select_node(node.children[0]) and
                node.children[1].predicate in ["GROUP_count"] and
                len(node.children[1].children) == 2 and
                node.children[1].children[0].predicate == "SELECT" and
                is_project_select_node(node.children[1].children[1]) and
                node.children[2].predicate == "GET_QUESTION_SPAN"):
            # Creating SELECT node
            project_string_arg = node.children[0].children[0].string_arg
            select_node = Node(predicate="SELECT")
            # original project arg1 is new select string-arg
            select_arg_node = Node(predicate="GET_QUESTION_SPAN", string_arg=project_string_arg)
            select_node.add_child(select_arg_node)
            # Creating ['PARTIAL_GROUP_count', ['PARTIAL_PROJECT', GET_QUESTION_SPAN('project_string_arg')]]
            partial_project_string_arg = node.children[1].children[0].children[0].string_arg
            partial_node = create_partial_group_node(count_or_sum="count",
                                                     project_string_arg=partial_project_string_arg)
            # Creating Condition node
            condition_string_arg = node.children[2].string_arg
            condition_node = Node(predicate="CONDITION")
            condition_string_arg_node = Node(predicate="GET_QUESTION_SPAN", string_arg=condition_string_arg)
            condition_node.add_child(condition_string_arg_node)
            # Replacing COMPARATIVE children to new ones made above
            new_children = [select_node, partial_node, condition_node]
            node.children = []
            for c in new_children:
                node.add_child(c)

        # CASE 3 --
        # ['COMPARATIVE',
        #   ['SELECT', 'field goals of Kris Brown'],
        #   ['PROJECT', 'yards of #REF', ['SELECT', 'field goals of Kris Brown']],
        #   'is at least 25']
        # ]
        # -->
        # ['COMPARATIVE',
        #   ['SELECT', 'field goals of Kris Brown'],
        #   ['PARTIAL_GROUP_sum', ['PARTIAL_SELECT_NUM', 'people of #REF']],
        #   ['CONDITION', 'is at least 25']
        # ]
        # Blatant use of PARTIAL_SELECT_NUM can be incorrect. e.g. "In which quarters did only the Panther's score?"
        # the project in annotation is -- ['PROJECT', 'who scores in #REF', ['SELECT', 'quarters']]
        elif (len(node.children) == 3 and
                node.children[0].predicate == "SELECT" and
                is_project_select_node(node.children[1]) and
                node.children[2].predicate == "GET_QUESTION_SPAN"):
            # Creating SELECT node
            select_node = node.children[0]
            # Creating ['PARTIAL_GROUP_sum', ['PARTIAL_SELECT_NUM', GET_QUESTION_SPAN('project_string_arg')]]
            project_string_arg = node.children[1].children[0].string_arg
            partial_node = create_partial_group_node(count_or_sum="sum", project_string_arg=project_string_arg)
            # Creating Condition node
            condition_string_arg = node.children[2].string_arg
            condition_node = Node(predicate="CONDITION")
            condition_string_arg_node = Node(predicate="GET_QUESTION_SPAN", string_arg=condition_string_arg)
            condition_node.add_child(condition_string_arg_node)
            # Replacing COMPARATIVE children to new ones made above
            new_children = [select_node, partial_node, condition_node]
            node.children = []
            for c in new_children:
                node.add_child(c)
        else:
            # 18 cases left in QDMR-high-level/DROP/train.json
            print("\n----COMPARATIVE----")
            print(question)
            print(node.get_nested_expression_with_strings())
            print()

    elif node.predicate in ["SUPERLATIVE_max", "SUPERLATIVE_min"]:
        # CASE 1
        # ['SUPERLATIVE_max',
        #   ['SELECT', 'quarterbacks'],
        #   ['GROUP_count',
        #     ['PROJECT', 'touchdown passes of #REF', ['SELECT', 'quarterbacks']],
        #     ['SELECT', 'quarterbacks']]
        # ]
        # -->
        # ['SUPERLATIVE_max',
        #   ['SELECT', 'quarterbacks'],
        #   ['PARTIAL_GROUP_count', ['PARTIAL_PROJECT', 'touchdown passes of #REF']]
        # ]
        if (len(node.children) == 2 and
                node.children[0].predicate == "SELECT" and
                node.children[1].predicate == "GROUP_count" and
                is_project_select_node(node.children[1].children[0])):
            select_string_arg = node.children[0].children[0].string_arg
            # Creating SELECT node
            select_node = node.children[0]
            # Creating ['PARTIAL_GROUP_count', ['PARTIAL_PROJECT', GET_QUESTION_SPAN('project_string_arg')]]
            partial_project_string_arg = node.children[1].children[0].children[0].string_arg
            partial_node = create_partial_group_node(count_or_sum="count",
                                                     project_string_arg=partial_project_string_arg)
            # Replacing SUPERLATIVE children to new ones made above
            new_children = [select_node, partial_node]
            node.children = []
            for c in new_children:
                node.add_child(c)
            lisp = nested_expression_to_lisp(node.get_nested_expression())
            qdmr_langugage.logical_form_to_action_sequence(lisp)
        # CASE 2
        # ['SUPERLATIVE_max',
        #   ['SELECT', 'kickers'],
        #   ['GROUP_sum',
        #     ['PROJECT', 'yards of #REF', ['SELECT', 'kickers']],
        #     ['SELECT', 'kickers']]
        # ]
        # -->
        # ['SUPERLATIVE_max',
        #   ['SELECT', 'kickers'],
        #   ['PARTIAL_GROUP_sum', 'SELECT_NUM_SPAN']
        # ]
        elif (len(node.children) == 2 and
                node.children[0].predicate == "SELECT" and
                node.children[1].predicate == "GROUP_sum" and
                is_project_select_node(node.children[1].children[0])):
            select_string_arg = node.children[0].children[0].string_arg
            # Creating SELECT node
            select_node = node.children[0]
            # Creating ['PARTIAL_GROUP_sum', ['PARTIAL_SELECT_NUM', GET_QUESTION_SPAN('project_string_arg')]]
            project_string_arg = node.children[1].children[0].children[0].string_arg
            partial_node = create_partial_group_node(count_or_sum="sum", project_string_arg=project_string_arg)
            # Replacing SUPERLATIVE children to new ones made above
            new_children = [select_node, partial_node]
            node.children = []
            for c in new_children:
                node.add_child(c)
            lisp = nested_expression_to_lisp(node.get_nested_expression())
            qdmr_langugage.logical_form_to_action_sequence(lisp)

    # Putting low-level operators down the if/else ladder so that if higher-order combinations are given priority
    # If node is one of ARITHMETIC_sum, ARITHMETIC_difference, ARITHMETIC_divison, ARITHMETIC_multiplication; then
    # both arguments need to be numbers. So if
    # (a) SELECT --> SELECT_NUM
    # (b) AGGREGATE_max --> SELECT_NUM(AGGREGATE_max)
    # (c) AGGREGATE_min --> SELECT_NUM(AGGREGATE_min)
    elif node.predicate in ["ARITHMETIC_sum", "ARITHMETIC_difference",
                            "ARITHMETIC_divison", "ARITHMETIC_multiplication"]:
        # assert len(node.children) == 2, f"These functions should only have two children : {node.get_nested_expression()}"
        new_children = []
        for child in node.children:
            if child.predicate == "SELECT":
                new_child = Node(predicate="SELECT_NUM")
                new_child.add_child(child)
                new_children.append(new_child)
            elif child.predicate == "AGGREGATE_max":
                new_child = Node(predicate="SELECT_NUM")
                new_child.add_child(child)
                new_children.append(new_child)
            elif child.predicate == "AGGREGATE_min":
                new_child = Node(predicate="SELECT_NUM")
                new_child.add_child(child)
                new_children.append(new_child)
            elif child.predicate == "PROJECT":
                new_child = Node(predicate="SELECT_NUM")
                new_child.add_child(child)
                new_children.append(new_child)
            else:
                new_children.append(child)
        # Replace old children with new ones
        node.children = []
        for c in new_children:
            node.add_child(c)

    if not skip_recursive_type_conforming:   # No condition sets this to True yet
        # Type-conforming all children of this node
        new_children = []
        for child in node.children:
            type_conforming_child = transform_to_fit_grammar(child, question)
            new_children.append(type_conforming_child)
        node.children = []
        for c in new_children:
            node.add_child(c)

    return node


def parse_qdmr_program_into_language(question: str, nested_expression: List):
    global template2questionprogram
    global template2count
    global total_not_parsed
    """Parse a QDMR decomposition represented as nested_expression into a program from our QDMRLanguage grammar.

    At the least, convert the string-arguments into STRING predicate.

    Transform certain predicates so that the resultant program is grammar-constrained.

    Find issues for programs that cannot be converted, define new predicates, etc.
    """
    # These gen-1 nested_expression contain string-arg as it is without a predicate. There is no notion predicates and
    # grammar yet. This script introduces a grammar for the first time.
    program_tree = nested_expression_to_tree(nested_expression, predicates_with_strings=False)
    # Wrap all string arguments with GET_QUESTION_SPAN predicate and move the string argument to node.string_arg
    program_tree = string_arg_to_quesspan_pred(node=program_tree)

    program_tree = transform_to_fit_grammar(program_tree, question, top_level=True)
    lisp_program = nested_expression_to_lisp(program_tree.get_nested_expression())

    try:
        qdmr_langugage.logical_form_to_action_sequence(lisp_program)
        return True, program_tree
    except:
        template = nested_expression_to_lisp(program_tree.get_nested_expression())
        program = program_tree.get_nested_expression_with_strings()
        if template not in template2questionprogram:
            template2questionprogram[template] = []
            template2count[template] = 0
        template2questionprogram[template].append((question, program))
        template2count[template] += 1
        total_not_parsed += 1

        return False, program_tree


def parse_qdmr_into_language(qdmr_examples: List[QDMRExample]) -> List[QDMRExample]:
    total_examples = len(qdmr_examples)
    total_examples_with_programs = 0
    type_conformed_programs = 0
    for qdmr_example in qdmr_examples:
        typed_nested_expression = []
        if len(qdmr_example.nested_expression):
            total_examples_with_programs += 1
            # empty nested_expression implies parsing error
            success, program_tree = parse_qdmr_program_into_language(qdmr_example.question,
                                                                     qdmr_example.nested_expression)
            if success:
                # nested_expr where string-arg nodes are written as PREDICATE(string-arg)
                typed_nested_expression = program_tree.get_nested_expression_with_strings()
                type_conformed_programs += 1
            qdmr_example.typed_nested_expression = typed_nested_expression

    print(f"Total examples: {total_examples}. W/ Programs: {total_examples_with_programs}")
    print(f"Type-conforming programs: {type_conformed_programs}")

    print("Not parsed. Total:{}".format(total_not_parsed))
    print("Num of templates: {}".format(len(template2count)))

    template2count_sorted = sorted(template2count.items(), key=lambda x: x[1], reverse=True)
    print(template2count_sorted[0:5])

    for i in range(0, 10):
        template, count = template2count_sorted[i]
        print("\n{} : {}".format(template, count))
        # questionprogram_list = template2questionprogram[template]
        # for j in range(0, min(5, len(questionprogram_list))):
        #     print(questionprogram_list[j])

    return qdmr_examples


def main(args):
    qdmr_json_path = args.qdmr_json
    qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(qdmr_json_path)
    qdmr_examples: List[QDMRExample] = parse_qdmr_into_language(qdmr_examples)

    examples_as_json_dicts = [example.to_json() for example in qdmr_examples]

    with open(qdmr_json_path, 'w') as outf:
        json.dump(examples_as_json_dicts, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdmr_json")
    args = parser.parse_args()

    main(args)







