import json
from typing import List, Tuple, Set, Dict, Union, Any

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

    # def _get_nested_expression_with_strings(self):
    #     """ Nested expression where predicates are replaced with string_arg if present.
    #         Note: This nested expression is only used for human-readability and debugging.
    #         This cannot be converted to lisp notation and is obviously not a parsable program.
    #     """
    #     string_or_predicate = self.string_arg if self.string_arg is not None else self.predicate
    #     if not self.is_leaf():
    #         nested_expression = [string_or_predicate]
    #         for child in self.children:
    #             nested_expression.append(child._get_nested_expression_with_strings())
    #         return nested_expression
    #     else:
    #         return string_or_predicate

    def get_nested_expression_with_strings(self):
        """ Nested expression where predicates w/ string_arg are written as PREDICATE(string_arg)
            This is introduced with drop_language since there are multiple predicates that select string_arg from text.
            Therefore, if we just write the string-arg to data (as done previously with typed_nested_expression), it
            would not be sufficient that some node is not a predicate in the language and hence get_ques_span pred.
            We can write the output of this function in json data and parse it back by removing split on ( and )
        """
        node_name = self.predicate
        if self.string_arg is not None:
            # GET_QUESTION_NUMBER(25)
            node_name = node_name + "(" + self.string_arg + ")"
        if self.is_leaf():
            return node_name
        else:
            nested_expression = [node_name]
            for child in self.children:
                nested_expression.append(child.get_nested_expression_with_strings())
            return nested_expression

class QDMRExample(object):
    def __init__(self, q_decomp):
        self.query_id = q_decomp["question_id"]
        self.question = q_decomp["question_text"]
        self.split = q_decomp["split"]
        self.decomposition = q_decomp["decomposition"]
        self.program: List[str] = q_decomp["program"]
        self.nested_expression: List = q_decomp["nested_expression"]
        self.operators = q_decomp["operators"]
        # Filled by parse_dataset/qdmr_grammar_program.py if transformation to QDMR-language is successful
        # This contains string-args as it is
        self.typed_nested_expression: List = []
        if "typed_nested_expression" in q_decomp:
            self.typed_nested_expression = q_decomp["typed_nested_expression"]

        # This was added after completing parse_dataset/drop_grammar_program. This class is a moving target.
        # This nested_expression should be s.t. string-arg grounding predicates occur as PREDICATE(string-arg).
        # e.g. ['FILTER_NUM_EQ', ['SELECT', 'GET_QUESTION_SPAN(field goals of Mason)'], 'GET_QUESTION_NUMBER(37)']
        self.drop_nested_expression: List = []
        if "drop_nested_expression" in q_decomp:
            self.drop_nested_expression = q_decomp["drop_nested_expression"]

        self.program_tree: Node = None
        self.typed_masked_nested_expr = []
        if self.drop_nested_expression:
            self.program_tree: Node = nested_expression_to_tree(self.drop_nested_expression,
                                                                predicates_with_strings=True)
            # This contains string-args masked as GET_QUESTION_SPAN predicate
            self.typed_masked_nested_expr = self.program_tree.get_nested_expression()

        elif self.typed_nested_expression:
            self.program_tree: Node = nested_expression_to_tree(self.typed_nested_expression,
                                                                predicates_with_strings=True)
            # This contains string-args masked as GET_QUESTION_SPAN predicate
            self.typed_masked_nested_expr = self.program_tree.get_nested_expression()

        self.extras: Dict = q_decomp.get("extras", {})

    def to_json(self):
        json_dict = {
            "question_id": self.query_id,
            "question_text": self.question,
            "split": self.split,
            "decomposition": self.decomposition,
            "program": self.program,
            "nested_expression": self.nested_expression,
            "typed_nested_expression": self.typed_nested_expression,
            "drop_nested_expression": self.drop_nested_expression,
            "operators": self.operators,
            "extras": self.extras,
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


def nested_expression_to_lisp(nested_expression) -> str:
    if isinstance(nested_expression, str):
        return nested_expression

    elif isinstance(nested_expression, List):
        lisp_expressions = [nested_expression_to_lisp(x) for x in nested_expression]
        return "(" + " ".join(lisp_expressions) + ")"
    else:
        raise NotImplementedError


def convert_nestedexpr_to_tuple(nested_expression):
    """Converts a nested expression list into a nested expression tuple to make the program hashable."""
    new_nested = []
    for i, argument in enumerate(nested_expression):
        if i == 0:
            new_nested.append(argument)
        else:
            if isinstance(argument, list):
                tupled_nested = convert_nestedexpr_to_tuple(argument)
                new_nested.append(tupled_nested)
            else:
                new_nested.append(argument)
    return tuple(new_nested)


def linearize_nested_expression(nested_expression, open_bracket: str = "(",
                                close_bracket: str = ")") -> List[str]:
    """Convert the program (as nested expression) into a linearized expression.

        The natural language arguments in the program are kept intact as a single program `token` and it is the onus of
        the processing step after this to tokenize them
    """
    if isinstance(nested_expression, str):
        # If the string is not a predicate but a NL argument, it is the onus of the models to tokenize it appropriately
        return [nested_expression]

    elif isinstance(nested_expression, List):
        # Flattened list of tokens for each element in the list
        program_tokens = []
        for x in nested_expression:
            program_tokens.extend(linearize_nested_expression(x, open_bracket, close_bracket))
        # Inserting a bracket around the program tokens
        program_tokens.insert(0, open_bracket)
        program_tokens.append(close_bracket)
        return program_tokens
    else:
        raise NotImplementedError


def nested_expression_to_tree(nested_expression, predicates_with_strings) -> Node:
    """ There are two types of expressions, one which have string-arg as it is and without a predicate (True)
    and other, newer DROP style, where string-arg nodes in expression are PREDICATE(string-arg)
    """
    if isinstance(nested_expression, str):
        if not predicates_with_strings:
            current_node = Node(predicate=nested_expression)
        else:
            predicate_w_stringarg = nested_expression
            # This can either be a plain predicate (e.g. `SELECT`) or with a string-arg (e.g. `GET_Q_SPAN(string-arg)`)
            start_paranthesis_index = predicate_w_stringarg.find("(")
            if start_paranthesis_index == -1:
                predicate = predicate_w_stringarg
                current_node = Node(predicate=predicate)
            else:
                predicate = predicate_w_stringarg[0:start_paranthesis_index]
                # +1 to avoid (, and -1 to avoid )
                string_arg = predicate_w_stringarg[start_paranthesis_index+1:-1]
                current_node = Node(predicate=predicate, string_arg=string_arg)

    elif isinstance(nested_expression, list):
        current_node = Node(nested_expression[0])
        for i in range(1, len(nested_expression)):
            child_node = nested_expression_to_tree(nested_expression[i], predicates_with_strings)
            current_node.add_child(child_node)
    else:
        raise NotImplementedError

    return current_node

def get_inorder_function_list(node: Node) -> List[str]:
    inorder_func_list = [node.predicate]
    for c in node.children:
        inorder_func_list.extend(get_inorder_function_list(c))
    return inorder_func_list


def get_inorder_function_list_from_template(tempalte: Union[Tuple, str]) -> List[str]:
    if isinstance(tempalte, tuple):
        inorder_func_list = [tempalte[0]]
        for c in tempalte[1:]:
            inorder_func_list.extend(get_inorder_function_list_from_template(c))
    else:
        return [tempalte]
    return inorder_func_list


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


def read_drop_dataset(input_json: str):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def convert_answer(answer_annotation: Dict[str, Union[str, Dict, List]]) -> Tuple[str, List]:
    answer_type = None
    if answer_annotation["spans"]:
        answer_type = "spans"
    elif answer_annotation["number"]:
        answer_type = "number"
    elif any(answer_annotation["date"].values()):
        answer_type = "date"

    answer_content = answer_annotation[answer_type] if answer_type is not None else None

    answer_texts = []
    if answer_type is None:  # No answer
        return None
    elif answer_type == "spans":
        # answer_content is a list of string in this case
        answer_texts = answer_content
    elif answer_type == "date":
        # answer_content is a dict with "month", "day", "year" as the keys
        date_tokens = [
            answer_content[key] for key in ["month", "day", "year"] if key in answer_content and answer_content[key]
        ]
        answer_texts = date_tokens
    elif answer_type == "number":
        # answer_content is a string of number
        answer_texts = [answer_content]
    return answer_type, answer_texts


if __name__ == "__main__":
    p = ['FILTER_NUM_GT', ['FILTER', ['SELECT', 'GET_QUESTION_SPAN(yards of TD passes)'], 'GET_QUESTION_SPAN(in the first half)'], 'GET_QUESTION_NUMBER(70)']

    node: Node = nested_expression_to_tree(p)
    print(node._get_nested_expression_with_predicate_and_strings())
    print(node.get_nested_expression_with_strings())
    print(node.get_nested_expression())