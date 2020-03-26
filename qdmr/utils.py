import json
from typing import List, Tuple, Set

from qdmr.domain_languages.qdmr_language import QDMRLanguage

qdmr_langugage = QDMRLanguage()
QDMR_predicates = list(qdmr_langugage._functions.keys())



class Node(object):
    def __init__(self, predicate, string_arg=None):
        self.predicate = predicate
        self.string_arg = string_arg
        # Empty list indicates leaf node
        self.children: List[Node] = []
        # parent==None indicates root
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
            nested_expression = [self.predicate]
            for child in self.children:
                nested_expression.append(child.get_nested_expression())
            return nested_expression
        else:
            return self.predicate

    def _get_nested_expression_with_strings(self):
        """This nested expression is only used for human-readability and debugging. This is not a parsable program"""
        string_or_predicate = self.string_arg if self.string_arg is not None else self.predicate
        if not self.is_leaf():
            nested_expression = [string_or_predicate]
            for child in self.children:
                nested_expression.append(child._get_nested_expression_with_strings())
            return nested_expression
        else:
            return string_or_predicate


class QDMRExample(object):
    def __init__(self, q_decomp):
        self.query_id = q_decomp["question_id"]
        self.question = q_decomp["question_text"]
        self.program: List[str] = q_decomp["program"]
        self.nested_expression: List = q_decomp["nested_expression"]
        self.operators = q_decomp["operators"]
        # Filled by parse_dataset/qdmr_grammar_program.py if transformation to QDMR-language is successful
        self.typed_nested_expression: List = []
        if "typed_nested_expression" in q_decomp:
            self.typed_nested_expression = q_decomp["typed_nested_expression"]
        self.program_tree: Node = None
        self.typed_masked_nested_expr = []
        if self.typed_nested_expression:
            self.program_tree = string_arg_to_quesspan_pred(nested_expression_to_tree(self.typed_nested_expression))
            self.typed_masked_nested_expr = self.program_tree.get_nested_expression()

    def to_json(self):
        json_dict = {
            "question_id": self.query_id,
            "question_text": self.question,
            "program": self.program,
            "nested_expression": self.nested_expression,
            "typed_nested_expression": self.typed_nested_expression,
            "operators": self.operators
        }
        return json_dict


def read_qdmr_json_to_examples(qdmr_json: str) -> List[QDMRExample]:
    """Parse processed qdmr json (from parse_dataset/parse_qdmr.py or qdmr_grammar_program.py into List[QDMRExample]"""
    qdmr_examples = []
    with open(qdmr_json, 'r') as f:
        dataset = json.load(f)
    for q_decomp in dataset:
        qdmr_example = QDMRExample(q_decomp)
        qdmr_examples.append(qdmr_example)
    return qdmr_examples


def write_qdmr_examples_to_json(qdmr_examples: List[QDMRExample], qdmr_json: str):
    examples_as_json_dicts = [example.to_json() for example in qdmr_examples]
    with open(qdmr_json, 'w') as outf:
        json.dump(examples_as_json_dicts, outf, indent=4)


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
        current_node = Node(predicate=nested_expression)

    elif isinstance(nested_expression, list):
        current_node = Node(nested_expression[0])
        for i in range(1, len(nested_expression)):
            child_node = nested_expression_to_tree(nested_expression[i])
            current_node.add_child(child_node)
    else:
        raise NotImplementedError

    return current_node


def string_arg_to_quesspan_pred(node: Node):
    """Convert ques-string arguments to functions in QDMR to generic STRING() function."""
    if node.predicate not in QDMR_predicates:
        node.string_arg = node.predicate
        node.predicate = "GET_QUESTION_SPAN"
    for child in node.children:
        string_arg_to_quesspan_pred(child)
    return node


def convert_nestedexpr_to_tuple(nested_expression) -> Tuple[Set[str], Tuple]:
    """Converts a nested expression list into a nested expression tuple to make the program hashable."""
    function_names = set()
    new_nested = []
    for i, argument in enumerate(nested_expression):
        if i == 0:
            function_names.add(argument)
            new_nested.append(argument)
        else:
            if isinstance(argument, list):
                func_set, tupled_nested = convert_nestedexpr_to_tuple(argument)
                function_names.update(func_set)
                new_nested.append(tupled_nested)
            else:
                new_nested.append(argument)
    return function_names, tuple(new_nested)