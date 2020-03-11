import json
from typing import List

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


def read_qdmr_json_to_examples(qdmr_json: str) -> List[QDMRExample]:
    qdmr_examples = []
    with open(qdmr_json, 'r') as f:
        dataset = json.load(f)
    for q_decomp in dataset:
        qdmr_example = QDMRExample(q_decomp)
        qdmr_examples.append(qdmr_example)
    return qdmr_examples