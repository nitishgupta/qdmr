from typing import List, Dict
import json
import re
import os
import argparse

from qdmr.domain_languages.qdmr_language import QDMRLanguage
from qdmr.domain_languages.drop_language import DROPLanguage
from qdmr.data.utils import Node, QDMRExample, nested_expression_to_lisp, nested_expression_to_tree, \
    string_arg_to_quesspan_pred, read_qdmr_json_to_examples, read_drop_dataset, convert_answer, \
    convert_nestedexpr_to_tuple

"""
This script is supposed to take QDMR-grammar-programs data (output from qdmr_grammar_program.py) as input and convert
programs to DROPLanguage according to analysis in 
https://docs.google.com/document/d/11OVeaEbqKvR-2lYKSS9jhHykANZDWGj9wObrWBuCZlE/edit
and
https://docs.google.com/spreadsheets/d/1Nu_4GqZhmYlD6D9PWe99M0_YttSsN-KNxy5OYrFxFkY/edit 
"""


qdmr_langugage = QDMRLanguage()
drop_language = DROPLanguage()
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


def get_nested_expr_tuple(qdmr_node: Node):
    nested_expr = qdmr_node.get_nested_expression()
    _, nested_expr_tuple = convert_nestedexpr_to_tuple(nested_expr)
    return nested_expr_tuple


def year_diff_single_event(qdmr_node: Node, question: str):
    """Convert AGGREGATE_count(SELECT(QSPAN("years that ... "))) into Year_Diff_Single_Event(SELECT) """
    nested_expr_tuple = get_nested_expr_tuple(qdmr_node)
    change = 0
    if nested_expr_tuple == ("AGGREGATE_count", ("SELECT", "GET_QUESTION_SPAN")):
        select_string_arg = qdmr_node.children[0].children[0].string_arg
        if "years" in select_string_arg:
            qdmr_node.predicate = "Year_Diff_Single_Event"
            change = 1

    return qdmr_node, change


def year_diff_two_events(qdmr_node: Node, question: str):
    """
        Convert ARITHMETIC_difference(SELECT_NUM(SELECT(QSPAN("year ..."), SELECT_NUM(SELECT(QSPAN("year ...")))) into
        Year_Diff_Two_Events(SELECT(QSPAN), SELECT(QSPAN))
    """

    def is_node_select_num_select_year(node: Node):
        matches_condition = False
        if node.predicate == "SELECT_NUM" and len(node.children) == 1:
            child1: Node = node.children[0]
            if child1.predicate == "SELECT" and len(child1.children) == 1:
                child2: Node = child1.children[0]
                if child2.predicate == "GET_QUESTION_SPAN":
                    if "year" in child2.string_arg:
                        matches_condition = True
        return matches_condition

    matches = False
    change = 0
    if qdmr_node.predicate == "ARITHMETIC_difference":
        if len(qdmr_node.children) == 2:
            child1: Node = qdmr_node.children[0]
            child2: Node = qdmr_node.children[0]
            if is_node_select_num_select_year(child1) and is_node_select_num_select_year(child2):
                matches = True

    if matches:
        year_diff_two_events_node = Node(predicate="Year_Diff_Two_Events")
        for select_num_node in qdmr_node.children:
            select_node = select_num_node.children[0]
            year_diff_two_events_node.add_child(select_node)
        qdmr_node = year_diff_two_events_node
        change = 1

    return qdmr_node, change


def compare_num_to_date(qdmr_node: Node, question: str):
    """ Convert
        COMPARISON_max/min(SELECT(QSPAN("when ...")), SELECT(QSPAN("when ..."))) into
        COMPARISON_DATE_min(SELECT(QSPAN), SELECT(QSPAN))
    """
    def is_node_select_when(node: Node):
        matches_condition = False
        if node.predicate == "SELECT" and len(node.children) == 1:
                child1: Node = node.children[0]
                if child1.predicate == "GET_QUESTION_SPAN":
                    if "when" in child1.string_arg:
                        matches_condition = True
        return matches_condition

    matches = False
    if qdmr_node.predicate in ["COMPARISON_max", "COMPARISON_min"]:
        if len(qdmr_node.children) == 2:
            child1: Node = qdmr_node.children[0]
            child2: Node = qdmr_node.children[0]
            if is_node_select_when(child1) and is_node_select_when(child2):
                matches = True

    if matches:
        root_predicate = "COMPARISON_DATE_min" if qdmr_node.predicate == "COMPARISON_min" else "COMPARISON_DATE_max"
        drop_node = Node(predicate=root_predicate)
        for child in qdmr_node.children:
            drop_node.add_child(child)
        return drop_node, 1

    return qdmr_node, 0


def filter_num_classifier(string_arg: str):
    """ Returns if a QSPAN string matches any of the FILTER_NUM_COND. If yes, finds the COND and the question-number
        Returns:
            matches: bool denoting if the qspan matched any patterns
            filter_type: one of "LT", "GT", "EQ", "LT_EQ", "GT_EQ".
            number: The number extracted from the input qspan string for the condition
    """

    patterns_for_gt = [
        "over [0-9]+",                                      # "over #NUM"
        "that (are|was|were) over [0-9]+$",                 # "that are over #NUM"
        "that (are|was|were) over [0-9]+( |\-)\w+$",        # "that are over #NUM yards"
        "that (are|was|were) longer than [0-9]+( |\-)\w+$", # "that are/was longer than 40 yards"
        "that (are|was|were) [0-9]+\+ \w+$",                 # "that are 40+ yard"
        "is (higher|longer) than [0-9]+$",
        "is (higher|longer) than [0-9]+( |\-)\w+$$",
    ]

    patterns_for_gt_eq = [
        "that (are|was|were) atleast [0-9]+$",  # "that are/was atleast 40"
        "that (are|was|were) atleast [0-9]+( |\-)\w+$",  # "that are/was atleast 40 yards"
        "that (are|was|were) at least [0-9]+$",  # "that are/was at least 40"
        "that (are|was|were) at least [0-9]+( |\-)\w+$",  # "that are/was at least 40 yards"
        "where the size is atleast [0-9]+$",
        "where the size is atleast [0-9]+( |\-)\w+$",
        "where the size is at least [0-9]+$",
        "where the size is at least [0-9]+( |\-)\w+$",
        "is atleast [0-9]+$",
        "is atleast [0-9]+( |\-)\w+$",
        "is at least [0-9]+$",
        "is at least [0-9]+( |\-)\w+$",
        "atleast [0-9]+$",
        "atleast [0-9]+( |\-)\w+$$",
        "at least [0-9]+$",
        "at least [0-9]+( |\-)\w+$$",
    ]

    patterns_for_eq = [
        "equal to [0-9]+$",                                  # "equal to #NUM"
        "equal to [0-9]+( |\-)\w+$",                         # "equal to #NUM-yards"
        "that (are|was|were) equal to [0-9]+$",              # "that are equal to #NUM"
        "that (are|was|were) equal to [0-9]+( \-)\w+$$",     # "that are equal to #NUM yard"
        "that (are|was|were) [0-9]+$",                       # "that are 40"
        "that (are|was|were) [0-9]+( |\-)\w+$",              # "that are 40 yards"
    ]

    patterns_for_lt = [
        "under [0-9]+$",                                       # "under #NUM"
        "under [0-9]+( |\-)\w+$",                              # "under #NUM-yards"
        "that (are|was|were) under [0-9]+$",                   # "that are under #NUM"
        "that (are|was|were) under [0-9]+( |\-)\w+$",          # "that are under #NUM-yard"
        "that (are|was|were) less than [0-9]+$",               # "that are/was less than 40"
        "that (are|was|were) less than [0-9]+( |\-)\w+$",      # "that are/was less than 40 yards"
        "that (are|was|were) shorter than [0-9]+( |\-)\w+$",   # "that are/was shorter than 40 yards"
        "that (are|was|were) shorter than [0-9]+$",            # "that are/was shorter than 40"
        "is lower than [0-9]+$",                               # "is lower than #NUM"
        "less than [0-9]+$",
        "less than [0-9]+( |\-)\w+$$",
        "is less than [0-9]+$",
        "is less than [0-9]+( |\-)\w+$$",
    ]

    patterns_for_lt_eq = [
        "that (are|was|were) atmost [0-9]+$",  # "that are/was atmost 40"
        "that (are|was|were) atmost [0-9]+( |\-)\w+$",  # "that are/was atmost 40 yards"
        "that (are|was|were) at most [0-9]+$",  # "that are/was at most 40"
        "that (are|was|were) at most [0-9]+( |\-)\w+$",  # "that are/was at most 40 yards"
        "atmost [0-9]+$",
        "atmost [0-9]+( |\-)\w+$$",
        "at most [0-9]+$",
        "at most [0-9]+( |\-)\w+$$",
    ]

    regex_progs_gt = [re.compile(p) for p in patterns_for_gt]
    regex_progs_lt = [re.compile(p) for p in patterns_for_lt]
    regex_progs_gt_eq = [re.compile(p) for p in patterns_for_gt_eq]
    regex_progs_lt_eq = [re.compile(p) for p in patterns_for_lt_eq]
    regex_progs_eq = [re.compile(p) for p in patterns_for_eq]

    matches = False
    # One of ["EQ", "LT", "GT", "GT_EQ", "LT_EQ"]
    filter_type = None
    for regex in regex_progs_eq:
        if regex.match(string_arg) is not None:
            matches = True
            filter_type = "EQ"

    for regex in regex_progs_lt:
        if regex.match(string_arg) is not None:
            matches = True
            filter_type = "LT"

    for regex in regex_progs_gt:
        if regex.match(string_arg) is not None:
            matches = True
            filter_type = "GT"

    for regex in regex_progs_gt_eq:
        if regex.match(string_arg) is not None:
            matches = True
            filter_type = "GT_EQ"

    for regex in regex_progs_lt_eq:
        if regex.match(string_arg) is not None:
            matches = True
            filter_type = "LT_EQ"

    # Matches #NUM or #NUM+
    number_regex_prog = re.compile("[0-9]+(|\+)")
    number = None
    if matches:
        tokens = string_arg.split(" ")
        tokens = [x for t in tokens for x in t.split("-")]
        reversed_tokens = tokens[::-1]
        for t in reversed_tokens:
            if number_regex_prog.fullmatch(t) is not None:
                number = t
                break

    return matches, filter_type, number


# Total number of filters encountered, and total number of filters chenged to filter_num_OP amongst them
def filter_to_filternum(qdmr_node: Node, question: str):
    """ Recursively, convert FILTER(SET, QSPAN) node into
        FILTER_NUM_CONDITION(SET, GET_Q_NUM) if the QSPAN matches any of the pre-defined regexes

        CONDITION is one of LT, GT, LT_EQ, GT_EQ
    """
    change = 0
    if qdmr_node.predicate == "FILTER":
        qspan_node = qdmr_node.children[1]
        qspan_string_arg = qspan_node.string_arg
        matches, filter_type, number = filter_num_classifier(qspan_string_arg)
        if matches:
            change = 1
            qdmr_node.predicate = "FILTER_NUM_{}".format(filter_type)
            qdmr_node.children[1].predicate = "GET_QUESTION_NUMBER"
            # String arg number should be extracted from qspan_string_arg above
            qdmr_node.children[1].string_arg = number

    new_children = []
    for child in qdmr_node.children:
        new_child, x = filter_to_filternum(child, question)
        new_children.append(new_child)
        change = min(1, change + x)

    qdmr_node.children = []
    for c in new_children:
        qdmr_node.add_child(c)

    return qdmr_node, change


total_w_partial_sum = 0
total_w_partial_count = 0
partial_select_single_num = 0
def fix_partial_sum_count(qdmr_node: Node, question: str):
    """ Convert
        PARTIAL_GROUP_sum(PARTIAL_SELECT_NUM(QSPAN)) or
        PARTIAL_GROUP_count(PARTIAL_PROJECT(QSPAN)) --> PARTIAL_SELECT_SINGLE_NUMBER(QSPAN)

        Certain conditions for sum --
        (a) question shouldn't contain "total"
        (b) QSPAN shouldn't contain points

        Certain conditions for count --
        (a) QSPAN shouldn't contain field goal, touchdown, interception
    """
    global total_w_partial_count, total_w_partial_sum, partial_select_single_num
    change = 0
    if (qdmr_node.predicate == "PARTIAL_GROUP_sum" and len(qdmr_node.children) == 1 and
            qdmr_node.children[0].predicate == "PARTIAL_SELECT_NUM"):
        # TODO(nitish): False positives:
        #  Which quarterback threw for fewer passing yards?
        #  Which Packer quarterback had more yards?
        total_w_partial_sum += 1
        # This is the q_span node for PARTIAL_SELECT_NUM
        get_qspan_node = qdmr_node.children[0].children[0]
        qspan_string_arg = get_qspan_node.string_arg
        if not ("total" in question or "scored" in qspan_string_arg):
            qdmr_node.predicate = "PARTIAL_SELECT_SINGLE_NUM"
            qdmr_node.children = []
            qdmr_node.add_child(get_qspan_node)
            partial_select_single_num += 1
            change = 1

    if (qdmr_node.predicate == "PARTIAL_GROUP_count" and len(qdmr_node.children) == 1 and
            qdmr_node.children[0].predicate == "PARTIAL_PROJECT"):
        total_w_partial_count += 1
        # This is the q_span node for PARTIAL_SELECT_NUM
        get_qspan_node = qdmr_node.children[0].children[0]
        qspan_string_arg = get_qspan_node.string_arg
        if not (any(x in qspan_string_arg for x in ["touchdown", "field goal", "interceptions", "passes"])):
            qdmr_node.predicate = "PARTIAL_SELECT_SINGLE_NUM"
            qdmr_node.children = []
            qdmr_node.add_child(get_qspan_node)
            partial_select_single_num += 1
            change = 1

    new_children = []
    for child in qdmr_node.children:
        new_child, x = fix_partial_sum_count(child, question)
        new_children.append(new_child)
        change = min(1, change + x)

    qdmr_node.children = []
    for c in new_children:
        qdmr_node.add_child(c)

    return qdmr_node, change


total_comparative = 0
total_event_comparative = 0
def fix_comparative(qdmr_node: Node, question: str):
    """Fix COMPARATIVE(SELECT, PARTIAL_SELECT_SINGLE_NUM, CONDITION) where Select is over FG, TD, etc. events

    These are converted to FILTER(SELECT, COND_QSPAN) (or FILTER_NUM_COND) where Select is over the ques_span of
    select_num and select. The COND_QSPAN is the same as QSPAN of Condition.

    Notes:
        1. "yards of" from the beginning of the select qpan is removed since that is implicit
        2. Other similar programs (e.g. w/ PARTIAL_GROUP_sum(PARTIAL_PROJECT)) can also be converted but DROP train has
           a couple of examples only

    Note: Run `fix_partial_sum_count` before this
    """

    global total_comparative, total_event_comparative
    change = 0
    if qdmr_node.predicate == "COMPARATIVE" and qdmr_node.children[0].predicate == "SELECT":
        select_qspan_node = qdmr_node.children[0].children[0]
        select_qspan_string_arg = select_qspan_node.string_arg

        if any([x in select_qspan_string_arg for x in ["field goal", "touchdown", "passes", "interceptions"]]) and \
                not any([x in select_qspan_string_arg for x in ["who", "Who"]]):
            total_event_comparative += 1
            if qdmr_node.children[1].predicate == "PARTIAL_SELECT_SINGLE_NUM":
                select_num_qspan_arg: str = qdmr_node.children[1].children[0].string_arg
                condition_arg = qdmr_node.children[2].children[0].string_arg

                filter_node = Node(predicate="FILTER")
                # SELECT node
                select_node = Node("SELECT")
                select_qspan_string_arg = select_num_qspan_arg.replace("#REF", select_qspan_string_arg)
                if select_qspan_string_arg[0:9] == "yards of ":
                    select_qspan_string_arg = select_qspan_string_arg[9:]
                qspan_node = Node(predicate="GET_QUESTION_SPAN", string_arg=select_qspan_string_arg)
                select_node.add_child(qspan_node)
                # FILTER_CONDITION_NODE
                filter_cond_span_node = Node(predicate="GET_QUESTION_SPAN", string_arg=condition_arg)
                # COMPILE filter node
                filter_node.add_child(select_node)
                filter_node.add_child(filter_cond_span_node)
                filter_node, _, = filter_to_filternum(filter_node, question)
                qdmr_node = filter_node
                change = 1

    new_children = []
    for child in qdmr_node.children:
        new_child, x = fix_comparative(child, question)
        new_children.append(new_child)
        change = min(1, change + x)

    qdmr_node.children = []
    for c in new_children:
        qdmr_node.add_child(c)

    return qdmr_node, change


select_w_filter = 0
def decompose_select_filter(qdmr_node: Node, question: str):
    """ Recursively convert
        SELECT("X in the first half") and similar to FILTER(SELECT(X), "in the first half")
        Conditions  -- "in the first/second/.. quarter/half)
    """
    global select_w_filter
    change = 0
    compiled_regex = re.compile("[\w\s]+ in the (first|second|third|fourth) (quarter|half)$")
    if (qdmr_node.predicate == "SELECT" and len(qdmr_node.children) == 1 and
            qdmr_node.children[0].predicate == "GET_QUESTION_SPAN"):
        select_qspan_node = qdmr_node.children[0]
        qspan_string_arg = select_qspan_node.string_arg
        if compiled_regex.fullmatch(qspan_string_arg) and "points" not in qspan_string_arg:
            tokens = qspan_string_arg.split(" ")
            filter_arg = " ".join(tokens[-4:])
            select_arg = " ".join(tokens[:-4])
            filter_node = Node("FILTER")
            filter_qspan_node = Node("GET_QUESTION_SPAN", string_arg=filter_arg)
            select_qspan_node.string_arg = select_arg
            filter_node.add_child(qdmr_node)
            filter_node.add_child(filter_qspan_node)
            qdmr_node = filter_node
            change = 1
            select_w_filter += 1

    new_children = []
    for child in qdmr_node.children:
        new_child, x = decompose_select_filter(child, question)
        new_children.append(new_child)
        change = min(1, change + x)

    qdmr_node.children = []
    for c in new_children:
        qdmr_node.add_child(c)

    return qdmr_node, change


def wrap_aggregate_minmax_w_selectnum(qdmr_node: Node, answer_type: str):
    """ Wrap find-min-num / find-max-num event with select-num if answer is a number. """
    if qdmr_node.predicate in ["AGGREGATE_min", "AGGREGATE_max"] and answer_type == "number":
        select_num_node = Node(predicate="SELECT_NUM")
        select_num_node.add_child(qdmr_node)
        return select_num_node, 1
    else:
        return qdmr_node, 0


def project_tranformations(qdmr_node: Node, question: str, answer_type: str):
    """ Conversions if root is PROJECT and the answer in a number
        1. question that need 100 - NUM are converted to DIFF(GET_IMPLICIT_NUM, SELECT_NUM)
        2. where project's qspan is "yards of #REF" are converted to SELECT-NUM
        3. "How many years" questions that actually are single event year diff
    """
    if answer_type is None:
        return qdmr_node, 0
    if qdmr_node.predicate == "PROJECT" and answer_type == "number":
        project_string_arg = qdmr_node.children[0].string_arg
        if project_string_arg in ["difference of 100 and #REF", "the difference of 100 and #REF",
                                  "the difference of 100 percent and #REF"]:
            # Input: PROJECT(GET_Q_SPAN, SELECT) needs to be  converted to
            #   ARITHMETIC_DIFF(SELECT_IMPLICIT_NUM, SELECT_NUM(SELECT))
            drop_node = Node(predicate="ARITHMETIC_difference")
            get_implicit_num = Node(predicate="SELECT_IMPLICIT_NUM", string_arg="100")
            select_num_node = Node(predicate="SELECT_NUM")
            select_node = qdmr_node.children[1]
            select_num_node.add_child(select_node)
            drop_node.add_child(get_implicit_num)
            drop_node.add_child(select_num_node)
            return drop_node, 1

        elif project_string_arg in ["yards of #REF", "the yards of #REF"]:
            select_num_node = Node(predicate="SELECT_NUM")
            # Adding SELECT node as child of SELECT_NUM
            select_num_node.add_child(qdmr_node.children[1])
            return select_num_node, 1

        elif "How many years" in question:
            question_tokens = question.split(" ")
            qspan_string_arg = " ".join(question_tokens[3:])  # Skip "How many years"
            qdmr_node = Node(predicate="Year_Diff_Single_Event")
            qspan_node = Node(predicate="GET_QUESTION_SPAN", string_arg=qspan_string_arg)
            select_node = Node(predicate="SELECT")
            select_node.add_child(qspan_node)
            qdmr_node.add_child(select_node)
            return qdmr_node, 1
        else:
            return qdmr_node, 0
    else:
        return qdmr_node, 0


def fix_comparison_sum_max(qdmr_node: Node, question: str):
    """ In COMPARISON_sum_max/min if the question compares points, these are actually num-lookup comparison"""
    change = 0
    if qdmr_node.predicate in ["COMPARISON_sum_max", "COMPARISON_sum_min"]:
        if "points" in question:
            predicate = "COMPARISON_max" if "max" in qdmr_node.predicate else "COMPARISON_min"
            qdmr_node.predicate = predicate
            change = 1
    return qdmr_node, change


def fix_aggregate_sum(qdmr_node: Node, question: str):
    """ AGGREGATE_SUM(SELECT) --> SELECT_NUM(SELECT) if either is true
         1. qspan is selecting points
         2. question has "total" in it. Like "total points scored by both teams"
    """
    change = 0
    if qdmr_node.predicate == "AGGREGATE_sum" and qdmr_node.children[0].predicate == "SELECT":
        select_node = qdmr_node.children[0]
        if len(select_node.children) == 1 and select_node.children[0].predicate == "GET_QUESTION_SPAN":
            # SUM(SELECT(QSPAN)) -- if QSPAN has points and question does not have total, change root to
            # SELECT_NUM
            qspan_string_arg = select_node.children[0].string_arg
            if "point" in qspan_string_arg and "total" not in question:
                qdmr_node.predicate = "SELECT_NUM"
                change = 1      # 63 / 347 changed in DROP_train
    return qdmr_node, change


######################


def is_td_event(string_arg: str):
    string_arg = string_arg.lower()
    td_event = False
    if (any(x in string_arg for x in ["touchdown", "field goal", "interceptions", "passes", "td"])):
        td_event = True
    return td_event


def aggregate_count_to_selectnum(qdmr_node: Node, question: str, answer_content: List[str]):
    """ Generate candidates by replace AGGREGATE_count -->  SELECT_NUM """
    change = 0
    number_answer_str = answer_content[0]
    try:
        answer_number = float(number_answer_str)
    except:
        return qdmr_node, change

    nested_expr = qdmr_node.get_nested_expression()
    _, nested_expr_tuple = convert_nestedexpr_to_tuple(nested_expr)


    if nested_expr_tuple == ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN')):
        # 996 questions: "point":28, td-events:437 (only 18 w/ ans>=10), non-td: 559- 128 w/ ans>=10
        # after seeing examples it looks like all questions w/ ans < 10 are count-based and others select-nm
        if answer_number >= 10:
            qdmr_node.predicate = "SELECT_NUM"
            change = 1

    if nested_expr_tuple == ('ARITHMETIC_difference',
                             ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN')),
                             ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN'))):
        # 334 questions. 208 w/ ans>10 -- count -> select-num. 126 w/ ans<10  -- 57 (both td; let it be) + 67 (non-td)
        if answer_number > 10:
            qdmr_node.children[0].predicate = "SELECT_NUM"
            qdmr_node.children[1].predicate = "SELECT_NUM"
            change = 1
        else:
            # total questions: 126
            select_str_arg_1 = qdmr_node.children[0].children[0].children[0].string_arg
            select_str_arg_2 = qdmr_node.children[1].children[0].children[0].string_arg
            if not is_td_event(select_str_arg_1) and not is_td_event(select_str_arg_2):
                # both non TD events = 67 questions
                qdmr_node.children[0].predicate = "SELECT_NUM"
                qdmr_node.children[1].predicate = "SELECT_NUM"
                change = 1

    if nested_expr_tuple == ('ARITHMETIC_sum',
                             ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN')),
                             ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN'))):
        # 64 questions. 52 q w/ ans > 15. All non-td events seem like require select-num
        select_str_arg_1 = qdmr_node.children[0].children[0].children[0].string_arg
        select_str_arg_2 = qdmr_node.children[1].children[0].children[0].string_arg
        if not is_td_event(select_str_arg_1) and not is_td_event(select_str_arg_2):
            qdmr_node.children[0].predicate = "SELECT_NUM"
            qdmr_node.children[1].predicate = "SELECT_NUM"
            change = 1

    if nested_expr_tuple == ('AGGREGATE_count', ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN')):
        # 147 questions. 131 questions w/ ans<10 -- all seem count ques.
        if answer_number >= 10:  # Only 16 questions but interesting. need diff(select, implicit_select)
            qdmr_node.predicate = "SELECT_NUM"
            change = 1

    if nested_expr_tuple == ('AGGREGATE_count', ('PROJECT', 'GET_QUESTION_SPAN', ('SELECT', 'GET_QUESTION_SPAN'))):
        # 42 questions. 16 questions w/ ans>=10.
        if answer_number >= 10:
            qdmr_node.predicate = "SELECT_NUM"
            change = 1

    return qdmr_node, change


def comparison_count_to_selectnum(qdmr_node: Node, question: str):
    """ Fix for programs with  COMPARISON_count_max / min """
    change = 0
    nested_expr = qdmr_node.get_nested_expression()
    _, nested_expr_tuple = convert_nestedexpr_to_tuple(nested_expr)

    if nested_expr_tuple in \
            [('COMPARISON_count_max', ('SELECT', 'GET_QUESTION_SPAN'), ('SELECT', 'GET_QUESTION_SPAN')),
             ('COMPARISON_count_min', ('SELECT', 'GET_QUESTION_SPAN'), ('SELECT', 'GET_QUESTION_SPAN'))]:
        # 340 questions. All are compare-count-minmax(select(qspan), select(qspan))
        # 59 questions w/ both td-events -- let it be
        select_str_arg_1 = qdmr_node.children[0].children[0].string_arg
        select_str_arg_2 = qdmr_node.children[1].children[0].string_arg
        # print(nested_expr_tuple)
        # print(qdmr_node._get_nested_expression_with_strings())
        if not is_td_event(select_str_arg_1) and not is_td_event(select_str_arg_2):
            # Seems all are select-num
            old_predicate = qdmr_node.predicate
            new_predicate = "COMPARISON_min" if "min" in old_predicate else "COMPARISON_max"
            qdmr_node.predicate = new_predicate
            change = 1

    return qdmr_node, change


def rename_predicates(qdmr_node: Node, question: str):
    predicate_mapping = {
        "AGGREGATE_min": "AGGREGATE_NUM_min",
        "AGGREGATE_max": "AGGREGATE_NUM_max",
        "COMPARISON_min": "COMPARISON_NUM_min",
        "COMPARISON_max": "COMPARISON_NUM_max",
        "SUPERLATIVE_max": "SUPERLATIVE_NUM_max",
        "SUPERLATIVE_min": "SUPERLATIVE_NUM_min",
    }
    current_predicate_name = qdmr_node.predicate
    new_predicate_name = predicate_mapping.get(current_predicate_name, current_predicate_name)
    qdmr_node.predicate = new_predicate_name

    new_children = []
    for child in qdmr_node.children:
        new_child = rename_predicates(child, question)
        new_children.append(new_child)
    qdmr_node.children = []
    for c in new_children:
        qdmr_node.add_child(c)
    return qdmr_node



def map_drop_question_to_answer_type(drop_dataset, split):
    queryid2anstype, queryid2anslist = {}, {}
    for passage_id, passage_info in drop_dataset.items():
        qa_pairs = passage_info["qa_pairs"]
        for qa_pair in qa_pairs:
            query_id = get_qdmr_query_id(passage_id, qa_pair["query_id"], split)
            answer_dict = qa_pair["answer"]
            # one of "spans", "date", "number"
            returns = convert_answer(answer_annotation=answer_dict)
            if returns:
                answer_type, answer_content = returns
                queryid2anstype[query_id] = answer_type
                queryid2anslist[query_id] = answer_content
    return queryid2anstype, queryid2anslist


def get_qdmr_query_id(passage_id, query_id, split):
    return "DROP_" + split + "_" + passage_id + "_" + query_id


def parse_qdmr_into_language(qdmr_examples: List[QDMRExample], queryid2anstype: Dict[str, str],
                             queryid2anscontent: Dict[str, List]) -> List[QDMRExample]:
    total_examples = len(qdmr_examples)
    total_examples_with_programs = 0
    type_conformed_programs = 0
    num_transformed_ques = 0
    num_ques_w_progs = 0
    num_ques = len(qdmr_examples)
    for qdmr_example in qdmr_examples:
        (query_id, question) = (qdmr_example.query_id, qdmr_example.question)
        answer_type = queryid2anstype[query_id]
        answer_content = queryid2anscontent.get(query_id, [""])
        program_tree = qdmr_example.program_tree
        if program_tree:
            (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13) = (0,0,0,0,0,0,0,0,0,0,0,0,0)
            num_ques_w_progs += 1

            # Here we will make the program go through a bunch of transformations. Each function returns a program-node
            # and a bool telling if it changed the original input.
            # 665 question transformed with the two year-tranformations
            transformed_tree, c1 = year_diff_single_event(program_tree, question)
            transformed_tree, c2 = year_diff_two_events(transformed_tree, question)
            # 416 questions w/ compare-num to compare-date
            transformed_tree, c3 = compare_num_to_date(transformed_tree, question)
            # 94 questions w/ partial_sum_count and comparative fix
            transformed_tree, c4 = fix_partial_sum_count(transformed_tree, question)
            transformed_tree, c5 = fix_comparative(transformed_tree, question)
            # 189 questions w/ select --> filter(select)
            transformed_tree, c6 = decompose_select_filter(transformed_tree, question)
            # 71 questions w/ filter-num
            transformed_tree, c7 = filter_to_filternum(transformed_tree, question)
            # # 60 questions w/ aggregate_minmax --> select-num(agg....
            transformed_tree, c8 = wrap_aggregate_minmax_w_selectnum(transformed_tree, answer_type=answer_type)
            # # 171 questions
            transformed_tree, c9 = project_tranformations(transformed_tree, question=question, answer_type=answer_type)
            # 86 questions w/ fix compare-sum and aggregate-sum
            transformed_tree, c10 = fix_comparison_sum_max(transformed_tree, question=question)
            transformed_tree, c11 = fix_aggregate_sum(transformed_tree, question=question)
            # 568 questions w/ aggregate_count -> select-num
            transformed_tree, c12 = aggregate_count_to_selectnum(transformed_tree, question,
                                                                 answer_content=answer_content)
            # 308 questions w/ compare-count to compare-num
            transformed_tree, c13 = comparison_count_to_selectnum(transformed_tree, question)

            change = min(1, c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13)

            try:
                # Make sure the (un)transformed program compiles
                transformed_nested_expr = transformed_tree.get_nested_expression()
                drop_language.logical_form_to_action_sequence(nested_expression_to_lisp(transformed_nested_expr))
            except:
                print(f"Transformation failed. qid: {query_id}")

            # This contains string-arg and is meant to written to file
            drop_nested_expression = transformed_tree.get_nested_expression_with_strings()
            qdmr_example.drop_nested_expression = drop_nested_expression

            if change:
                # print(qdmr_example.query_id)
                # print(qdmr_example.question)
                # print(transformed_tree._get_nested_expression_with_strings())
                # print(answer_content)
                num_transformed_ques += 1

    print(f"Number of questions: {num_ques}  Number of questions w/ programs: {num_ques_w_progs}")
    print(f"Num of question w/ program transformations: {num_transformed_ques}")

    return qdmr_examples


def main(args):
    qdmr_json_path = args.qdmr_json
    qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(qdmr_json_path)
    drop_data = read_drop_dataset(args.drop_json)
    queryid2anstype, queryid2anslist = map_drop_question_to_answer_type(drop_data, args.qdmr_split)

    """ COUNT TEMPLATES """
    # count_templates = {}
    # for qdmr_example in qdmr_examples:
    #     program_tree = qdmr_example.program_tree
    #     if program_tree:
    #         nested_expr = program_tree.get_nested_expression()
    #         lisp = nested_expression_to_lisp(nested_expr)
    #         if "COMPARISON_count_min" in lisp or "COMPARISON_count_max" in lisp:
    #             lisp_tuple = convert_nestedexpr_to_tuple(nested_expr)
    #             count_templates[lisp_tuple] = count_templates.get(lisp_tuple, 0) + 1
    #
    # sorted_dict = sorted(count_templates.items(), key=lambda x: x[1], reverse=True)
    # print("\n".join([f"{x} : {y}" for x, y in sorted_dict]))
    # print(len(count_templates))
    # exit()

    qdmr_examples: List[QDMRExample] = parse_qdmr_into_language(qdmr_examples, queryid2anstype, queryid2anslist)

    examples_as_json_dicts = [example.to_json() for example in qdmr_examples]

    output_json = args.output_json
    output_dir = os.path.split(output_json)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json, 'w') as outf:
        json.dump(examples_as_json_dicts, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdmr_json")
    parser.add_argument("--drop_json")
    parser.add_argument("--qdmr_split")
    parser.add_argument("--output_json")
    args = parser.parse_args()

    main(args)







