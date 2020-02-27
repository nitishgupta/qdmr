from typing import List, Dict, Tuple

import json
from collections import defaultdict
import random
import argparse

from allennlp_semparse.common.util import lisp_to_nested_expression
from qdmr.domain_languages.qdmr_language import QDMRLanguage


"""
*initial goal of this script; update after we actually implement this*
This script takes QDMR programs parsed using parse_dataset.parse_qdmr and uses the grammar proposed in QDMRLanguage
to see what programs can be produced from this grammar. This would require certain transformations due to quirks of the
QDMR annotation. Also, not every program can be successfully mapped to a typed-grammar. Let see ...
"""


QDMR_predicates = ["SELECT", "COMPARISON_max", "AGGREGATE_count", "COMPARISON_min", "ARITHMETIC_sum",
                   "ARITHMETIC_difference", "AGGREGATE_min", "PROJECT", "FILTER", "BOOLEAN", "COMPARATIVE",
                   "AGGREGATE_max", "UNION", "INTERSECTION", "DISCARD", "COMPARISON_true", "GROUP_count",
                   "AGGREGATE_sum", "ARITHMETIC_division", "SUPERLATIVE_min", "SUPERLATIVE_max", "AGGREGATE_avg",
                   "ARITHMETIC_multiplication", "GROUP_sum", "SELECT_NUM"]

qdmr_langugage = QDMRLanguage()
QDMR_predicates = list(qdmr_langugage._functions.keys())


class QDMRExample():
    def __init__(self, q_decomp):
        self.query_id = q_decomp["question_id"]
        self.question = q_decomp["question_text"]
        self.program: List[str] = q_decomp["program"]
        self.nested_expression: List = q_decomp["nested_expression"]
        self.operators = q_decomp["operators"]


class Node(object):
    def __init__(self, data):
        # if data not in QDMR_predicates:
        #     data = "STRING"
        self.data = data
        # Empty list indicates leaf node
        self.children: List[Node] = []
        # None parent indicates Root
        self.parent: Node = None

    def add_child(self, obj):
        assert isinstance(obj, Node)
        obj.parent = self
        self.children.append(obj)

    def is_leaf(self):
        leaf = True if not len(self.children) else False
        return leaf

    def get_nested_expression(self):
        if not self.is_leaf():
            nested_expression = [self.data]
            for child in self.children:
                nested_expression.append(child.get_nested_expression())
            return nested_expression
        else:
            return self.data


def nested_expression_to_lisp(nested_expression):
    if isinstance(nested_expression, str):
        return nested_expression

    elif isinstance(nested_expression, List):
        lisp_expressions = [nested_expression_to_lisp(x) for x in nested_expression]
        return "(" + " ".join(lisp_expressions) + ")"
    else:
        raise NotImplementedError



def nested_expression_to_tree(nested_expression) -> Node:
    if isinstance(nested_expression, str):
        current_node = Node(data=nested_expression)

    elif isinstance(nested_expression, List):
        current_node = Node(nested_expression[0])
        children_node = []
        for i in range(1, len(nested_expression)):
            child_node = nested_expression_to_tree(nested_expression[i])
            current_node.add_child(child_node)

    return current_node


def mask_string_argument_to_type(node: Node):
    """Convert ques-string arguments to functions in QDMR to generic STRING() function."""
    if node.data not in QDMR_predicates:
        node.data = "GET_QUESTION_SPAN"
    for child in node.children:
        mask_string_argument_to_type(child)
    return node

def convert_to_type_conforming(node: Node):
    """This function takes a program as a Tree, and converts predicates so the resulting program is type-conforming.

    QDMR, for example, could have a program such as, COMPARISON_max(SELECT, SELECT) which should ideally be
    COMPARISON_max(SELECT_NUM, SELECT_NUM)

    As a first step, we simply gather a list of predicates that require NUMBER arguments, and if their actual children
    are SELECT, we convert them to SELECT-NUM
    """

    # If node is one of ARITHMETIC_sum, ARITHMETIC_difference, ARITHMETIC_divison, ARITHMETIC_multiplication; then
    # both arguments need to be numbers. So if
    # (a) SELECT --> SELECT_NUM
    # (b) AGGREGATE_max --> SELECT_NUM(AGGREGATE_max)
    # (c) AGGREGATE_min --> SELECT_NUM(AGGREGATE_min)

    if node.data in ["ARITHMETIC_sum", "ARITHMETIC_difference", "ARITHMETIC_divison", "ARITHMETIC_multiplication"]:
        # assert len(node.children) == 2, f"These functions should only have two children : {node.get_nested_expression()}"
        new_children = []
        for child in node.children:
            if child.data == "SELECT":
                new_child = Node(data="SELECT_NUM")
                new_child.add_child(child)
                new_children.append(new_child)
            elif child.data == "AGGREGATE_max":
                new_child = Node(data="SELECT_NUM")
                new_child.add_child(child)
                new_children.append(new_child)
            elif child.data == "AGGREGATE_min":
                new_child = Node(data="SELECT_NUM")
                new_child.add_child(child)
                new_children.append(new_child)
            elif child.data == "PROJECT":
                new_child = Node(data="SELECT_NUM")
                new_child.add_child(child)
                new_children.append(new_child)
            else:
                new_children.append(child)
        # Replace old children with new ones
        node.children = []
        for c in new_children:
            node.add_child(c)

    # Trivia -- 300 programs in DROP_train were fixed by this!!
    elif node.data in ["COMPARISON_max", "COMPARISON_min"]:
        # First check if all children are AGGREGATE_count(SELECT)
        all_chilren_count_select = True
        for child in node.children:
            child_is_count_select = False
            if child.data == "AGGREGATE_count":
                if len(child.children) == 1 and child.children[0].data == "SELECT":
                    child_is_count_select = True
            all_chilren_count_select = all_chilren_count_select and child_is_count_select

        if all_chilren_count_select:
            # Need to remove AGGREGATE_count from all children and change the function at this node.
            new_children = []
            for child in node.children:
                # We know from the test above that the child is AGGREGATE_count and it only has one child that is SELECT
                new_child = child.children[0]
                new_children.append(new_child)
                # Replace old children with new ones
            node.children = []
            for c in new_children:
                node.add_child(c)
            node.data = "COMPARISON_count_max" if node.data == "COMPARISON_max" else "COMPARISON_count_min"

    elif node.data in ["COMPARATIVE"]:
        if node.children[1].data == "PROJECT":
            node.children[1].data = "GROUP_property"

    # Type-conforming all children of this node
    new_children = []
    for child in node.children:
        type_conforming_child = convert_to_type_conforming(child)
        new_children.append(type_conforming_child)
    node.children = []
    for c in new_children:
        node.add_child(c)

    return node


def parse_qdmr_program_into_language(question: str, nested_expression: List):
    """Parse a QDMR decomposition represented as nested_expression into a program from our QDMRLanguage grammar.

    At the least, convert the string-arguments into STRING predicate.

    Transform certain predicates so that the resultant program is grammar-constrained.

    Find issues for programs that cannot be converted, define new predicates, etc.
    """
    original_lisp = nested_expression_to_lisp(nested_expression)
    program_tree = nested_expression_to_tree(nested_expression)
    program_tree = mask_string_argument_to_type(node=program_tree)
    old_lisp_program = nested_expression_to_lisp(program_tree.get_nested_expression())

    program_tree = convert_to_type_conforming(program_tree)
    lisp_program = nested_expression_to_lisp(program_tree.get_nested_expression())

    try:
        qdmr_langugage.logical_form_to_action_sequence(lisp_program)
        return True

    except:
        print(question)
        print(original_lisp)
        print(lisp_program)
        print("\n")
        return False
        # # print("Program cannot be parsed")
        # # print(lisp_program)
        # typed_program_tree = convert_to_type_conforming(program_tree)
        # if typed_program_tree.data in ["ARITHMETIC_sum", "ARITHMETIC_difference", "ARITHMETIC_divison", "ARITHMETIC_multiplication"]:
        #     nes_exp = typed_program_tree.get_nested_expression()
        #     new_lisp_program = nested_expression_to_lisp(nes_exp)
        #     try:
        #         qdmr_langugage.logical_form_to_action_sequence(new_lisp_program)
        #         print(lisp_program)
        #         print(new_lisp_program)
        #         return True
        #         # print("Program cannot be parsed")
        #         # print(lisp_program)
        #         # print(new_lisp_program)
        #     except:
        #         return False




def parse_qdmr_into_language(qdmr_examples: List[QDMRExample]):
    total_examples = len(qdmr_examples)
    total_examples_with_programs = 0
    type_conformed_programs = 0
    for qdmr_example in qdmr_examples:
        if len(qdmr_example.nested_expression):
            total_examples_with_programs += 1
            # empty nested_expression implies parsing error
            success = parse_qdmr_program_into_language(qdmr_example.question, qdmr_example.nested_expression)
            if success:
                type_conformed_programs += 1

    print(f"Total examples: {total_examples}. W/ Programs: {total_examples_with_programs}")
    print(f"Type-conforming programs: {type_conformed_programs}")


def read_dataset(qdmr_json: str) -> List[QDMRExample]:
    qdmr_examples = []
    with open(qdmr_json, 'r') as f:
        dataset = json.load(f)
    for q_decomp in dataset:
        qdmr_example = QDMRExample(q_decomp)
        qdmr_examples.append(qdmr_example)
    return qdmr_examples



def main(args):

    # nested_expression = ['ARITHMETIC_difference', ['SELECT_NUM', ['AGGREGATE_max', ['SELECT', 'STRING']]], ['SELECT_NUM', ['AGGREGATE_min', ['SELECT', 'STRING']]]]
    #
    # node: Node = nested_expression_to_tree(nested_expression)
    # print(nested_expression)
    # print(node.get_nested_expression())
    # lisp_program = nested_expression_to_lisp(nested_expression)

    # lisp_program = "(AGGREGATE_count (COMPARATIVE (SELECT STRING) (PROJECT STRING (SELECT STRING)) STRING))"
    # progtree = nested_expression_to_tree(lisp_to_nested_expression(lisp_program))
    # new_progtree = convert_to_type_conforming(progtree)
    # new_lisp = nested_expression_to_lisp(new_progtree.get_nested_expression())
    # print(new_lisp)
    # print(qdmr_langugage.logical_form_to_action_sequence(new_lisp))

    qdmr_json = args.qdmr_json
    qdmr_examples: List[QDMRExample] = read_dataset(qdmr_json)
    parse_qdmr_into_language(qdmr_examples)

    # output_dir = os.path.split(qdmr_json)[0]
    # print(output_dir)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # convert_to_json(qdmr_csv, qdmr_json, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdmr_json")
    args = parser.parse_args()

    main(args)







