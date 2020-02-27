import json


drop_json = "/shared/nitishg/data/drop_post_iclr/date_num/date_yd_num_hmyw_cnt_whoarg/drop_dataset_train.json"
qdmr_json = "/shared/nitishg/data/break-dataset/QDMR-high-level/json/DROP_train.json"



def read_drop(drop_json):
    total_ques = 0
    total_super = 0

    with open(drop_json, 'r') as f:
        dataset = json.load(f)

    qid2sup = {}
    qid2ques = {}
    for para_id, passage_info in dataset.items():
        qa_pairs = passage_info["qa_pairs"]
        for qa in qa_pairs:
            query_id = qa["query_id"]
            question = qa["question"]
            query_id = "DROP_train_" + para_id + "_" + query_id
            program_supervised = False
            if "program_supervised" in qa:
                if qa["program_supervised"]:
                    program_supervised = True

            qid2sup[query_id] = program_supervised
            qid2ques[query_id] = question
            total_ques += 1
            if program_supervised:
                total_super += 1
    print("Total questions: {}  Total supervised: {}".format(total_ques, total_super))
    return qid2sup, qid2ques


def read_qdmr(qdmr_json):
    total_ques = 0
    total_super = 0

    with open(qdmr_json, 'r') as f:
        dataset = json.load(f)

    qid2sup = {}
    qid2ques = {}
    for q_decomp in dataset:
        query_id = q_decomp["question_id"]
        question = q_decomp["question_text"]
        program = q_decomp["program"]
        program_supervised = False
        if "None" not in program:
            program_supervised = True

        qid2sup[query_id] = program_supervised
        qid2ques[query_id] = question
        total_ques += 1
        if program_supervised:
            total_super += 1

    print("Total questions: {}  Total supervised: {}".format(total_ques, total_super))
    return qid2sup, qid2ques

drop_qid2sup, drop_qid2ques = read_drop(drop_json)
qdmr_qid2sup, qdmr_qid2ques = read_qdmr(qdmr_json)

drop_sup_qids = [qid for qid, sup in drop_qid2sup.items() if sup is True]
qdmr_sup_qids = [qid for qid, sup in qdmr_qid2sup.items() if sup is True]

drop_qids = set(drop_sup_qids)
qdmr_qids = set(qdmr_sup_qids)

drop_qdmr_common = drop_qids.intersection(qdmr_qids)
print("Common in drop and qdmr = {}".format(len(drop_qdmr_common)))

qdmr_extra = qdmr_qids.difference(drop_qids)
print("Extra in qdmr: {}".format(len(qdmr_extra)))

