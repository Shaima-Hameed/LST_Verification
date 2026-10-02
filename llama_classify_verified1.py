# =====================================================================================================
# Enhancing Logical Fallacy Detection through Semantic Relation Verification of Logical Structure Trees
#
# Based on the original Lei & Huang (EMNLP 2024) LST code. Edits are marked:
#   [NEW]      code that did not exist in the original
#   [CHANGED]  original code that was modified
#   [FIX]      corrections made after review, so the code matches the methodology
# Everything unmarked is the original code, unchanged.
#
# Stages (as in the methodology figure):
#   1. Logical structure tree construction        -> unchanged (only cached to disk so it runs once)
#   2. Semantic relation verification             -> [NEW] NLI verifier + LLM-as-a-judge, relation-specific
#   3. Verdict aggregation over the tree          -> [NEW] effective(n) = own(n) AND effective(children), post hoc flag
#   4. Verification-aware integration             -> [CHANGED] verdict column + tree summary (hard prompt),
#                                                    verification vector v in the relation encoders (soft prompt)
#   5. Classification and explanation             -> Llama-2-7B + LoRA label; [NEW] explanation assembled
#                                                    from failed relations + judge rationales (not generated)
#
# [FIX] Every experiment is selected from the command line, e.g.
#   python llama_classify_verified.py --mode both --seed 42              full model (NLI + judge)
#   python llama_classify_verified.py --mode nli --seed 1                NLI-only verifier
#   python llama_classify_verified.py --mode judge                       judge-only verifier
#   python llama_classify_verified.py --mode none                        CONTROL: same tree encoder and table format, no verification
#   python llama_classify_verified.py --mode both --no_vector            ablation: verdicts in the text prompt only
#   python llama_classify_verified.py --mode both --no_text              ablation: verdicts in the tree embedding only
#   python llama_classify_verified.py --mode both --shuffle              ablation: verdicts shuffled across relations
#   python llama_classify_verified.py --mode both --gated                gated encoder variant
#   python llama_classify_verified.py --mode both --smoke                quick end-to-end check on 50/20/20 examples, 1 epoch
# =====================================================================================================

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'


import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import RobertaTokenizer, RobertaModel
from peft import get_peft_model, LoraConfig, PeftConfig, PeftModel, PeftModelForSeq2SeqLM
import math
import random
import time
import datetime
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from transformers import get_linear_schedule_with_warmup
from scipy.optimize import linear_sum_assignment
from math import floor
from accelerate import Accelerator
import stanza
from nltk.tree import Tree, ParentedTree
import copy      # [NEW]
import gc        # [NEW]
import argparse  # [FIX]




''' [FIX] experiment switches (command line) '''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="both", choices=["none", "nli", "judge", "both"],
                    help="Stage 2 configuration; 'none' is the control without any verification")
parser.add_argument("--gated", action="store_true", help="Stage 4 gated encoder variant")
parser.add_argument("--no_text", action="store_true", help="ablation: no verdict column / tree summary in the hard prompt")
parser.add_argument("--no_vector", action="store_true", help="ablation: v set to zeros in the tree encoder")
parser.add_argument("--shuffle", action="store_true", help="ablation: shuffle verdicts across relations, then re-aggregate")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--smoke", action="store_true", help="quick check: 50/20/20 examples and 1 epoch")
parser.add_argument("--keep_checkpoint", action="store_true", help="keep the ~14GB checkpoint after testing")
args, _ = parser.parse_known_args()




''' hyper-parameters '''

dataset_name = "logic"
max_source_length = 1024 # argotario 476, logic 1068, political_debates 1013, reddit 1130, propaganda 1052
max_target_length = 256
no_decay = ['bias', 'layernorm.weight', 'LayerNorm.weight']
weight_decay = 1e-2
valid_steps = 512
batch_size = 1
num_epochs = 1 if args.smoke else args.epochs # [FIX]
gradient_accumulation_steps = 4
warmup_proportion = 0
roberta_lr = 1e-5
tree_lr = 2e-5
llama_lr = 3e-4
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05
seed_val = args.seed # [FIX] moved here so it can be part of the run name


# [NEW] verification hyper-parameters (Stages 2-4)
VERIFIER_MODE = args.mode                     # [FIX] "nli", "judge", "both" (combined), or "none" (control)
USE_GATED_ENCODER = args.gated                # Stage 4 gated variant: separate encoders for supported / unsupported relations
USE_VERDICT_TEXT = not args.no_text           # [FIX] verdict column + tree summary in the hard prompt
USE_VERDICT_VECTOR = not args.no_vector       # [FIX] verification vector v in the relation encoders
SHUFFLE_VERDICTS = args.shuffle               # [FIX] ablation: are the verdicts informative, or is it just extra input?
if VERIFIER_MODE == "none":                   # [FIX] control: same encoder and table format, but no verification information
    USE_VERDICT_TEXT = False
    USE_VERDICT_VECTOR = False
    SHUFFLE_VERDICTS = False

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"  # DeBERTa-v3-large NLI
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # open instruct model for LLM-as-a-judge (T = 0, greedy)
NLI_THRESHOLD = 0.5               # entailment probability for "supported" (to be calibrated on the hand-labeled set)
NLI_CIRCULAR_THRESHOLD = 0.8      # [FIX] entailment probability needed in BOTH directions for "circular"
NLI_BATCH_SIZE = 32
JUDGE_MAX_NEW_TOKENS = 60         # length of the one-sentence judge rationale
VERDICT_EMB_DIM = 16              # size of emb(verdict) in the verification vector v
V_DIM = 3 + 3 + VERDICT_EMB_DIM + 1 + 1  # v = [P_NLI(a->b) ; P_NLI(b->a) ; emb(verdict) ; conf ; flag]
CACHE_DIR = "./verification_cache/"      # Stage 1 trees, NLI scores and judge outputs are cached here (judge runs once offline)
OUTPUT_DIR = "./outputs/"

# [FIX] one name per experiment, used for the checkpoint, the predictions file and the results log
RUN_NAME = VERIFIER_MODE + ("_gated" if USE_GATED_ENCODER else "") + "_text" + str(int(USE_VERDICT_TEXT)) + \
           "_vec" + str(int(USE_VERDICT_VECTOR)) + ("_shuffled" if SHUFFLE_VERDICTS else "") + "_seed" + str(seed_val) + \
           ("_smoke" if args.smoke else "")
checkpoint_path = "./saved_models/llama_classify_verified_" + RUN_NAME + ".ckpt"  # [FIX]
print("Run:", RUN_NAME)






''' fallacy list and definition '''

if dataset_name == "argotario":
    fallacy_list = ['Ad Hominem', 'Emotional Language', 'Hasty Generalization', 'Irrelevant Authority', 'Red Herring']
    fallacy_def = ['Ad Hominem: the text attack a person instead of arguing against the claims.',
                   'Emotional Language: the text arouse non-rational emotions.',
                   'Hasty Generalization: the text draw a broad conclusion based on a limited sample of population.',
                   'Irrelevant Authority: the text cite an authority but the authority lacks relevant expertise.',
                   'Red Herring: the text diverge the attention to irrelevant issues.']

if dataset_name == "logic":
    fallacy_list = ['Ad Hominem', 'Ad Populum', 'Black-and-White Fallacy', 'False Cause', 'Circular Reasoning', 'Deductive Fallacy', 'Emotional Language',
                    'Equivocation', 'Extension Fallacy', 'Hasty Generalization', 'Intentional Fallacy', 'Irrelevant Authority', 'Red Herring']
    fallacy_def = ['Ad Hominem: the text attack a person instead of arguing against the claims.',
                   'Ad Populum: the text affirm something is true because the majority thinks so.',
                   'Black-and-White Fallacy: the text present two alternative options as the only possibilities.',
                   'False Cause: the text assume two correlated events must also have a causal relation.',
                   'Circular Reasoning: the end of the text come back to the beginning without having proven itself.',
                   'Deductive Fallacy: the text has an error in the logical reasoning.',
                   'Emotional Language: the text arouse non-rational emotions.',
                   'Equivocation: the text use a key term in multiple senses, leading to ambiguous conclusions.',
                   'Extension Fallacy: the text attack an exaggerated version of the opponent’s claim.',
                   'Hasty Generalization: the text draw a broad conclusion based on a limited sample of population.',
                   'Intentional Fallacy: the text show intentional action to incorrectly support an argument.',
                   'Irrelevant Authority: the text cite an authority but the authority lacks relevant expertise.',
                   'Red Herring: the text diverge the attention to irrelevant issues.']

if dataset_name == "political_debates":
    fallacy_list = ['Ad Hominem', 'Slippery Slope', 'Irrelevant Authority', 'Emotional Language', 'Slogans', 'False Cause']
    fallacy_def = ['Ad Hominem: the text attack a person instead of arguing against the claims.',
                   'Slippery Slope: the text suggest taking a small initial step leads to a chain of related events culminating in significant effect.',
                   'Irrelevant Authority: the text cite an authority but the authority lacks relevant expertise.',
                   'Emotional Language: the text arouse non-rational emotions.',
                   'Slogans: the text use a brief and striking phrase to provoke excitement of the audience.',
                   'False Cause: the text assume two correlated events must also have a causal relation.']

if dataset_name == "reddit":
    fallacy_list = ['Slippery Slope', 'Irrelevant Authority', 'Hasty Generalization', 'Black-and-White Fallacy',
                    'Ad Populum', 'Tradition Fallacy', 'Naturalistic Fallacy', 'Worse Problem Fallacy']
    fallacy_def = ['Slippery Slope: the text suggest taking a small initial step leads to a chain of related events culminating in significant effect.',
                   'Irrelevant Authority: the text cite an authority but the authority lacks relevant expertise.',
                   'Hasty Generalization: the text draw a broad conclusion based on a limited sample of population.',
                   'Black-and-White Fallacy: the text present two alternative options as the only possibilities.',
                   'Ad Populum: the text affirm something is true because the majority thinks so.',
                   'Tradition Fallacy: the text argue the action has always been done in the tradition.',
                   'Naturalistic Fallacy: the text claim something is good or bad because it is natural or unnatural.',
                   'Worse Problem Fallacy: the text justify an issue by arguing more severe issues exists.']

if dataset_name == "climate":
    fallacy_list = ['Evading Burden of Proof', 'Cherry Picking', 'Red Herring', 'Strawman', 'Irrelevant Authority',
                    'Hasty Generalization', 'False Cause', 'False Analogy', 'Vagueness']
    fallacy_def = ['Evading Burden of Proof: the text make a claim without evidence or supporting argument.',
                   'Cherry Picking: the text selectively present partial evidence to support a claim.',
                   'Red Herring: the text diverge the attention to irrelevant issues.',
                   'Strawman: the text distort the claim to another one to make it easier to attack.',
                   'Irrelevant Authority: the text cite an authority but the authority lacks relevant expertise.',
                   'Hasty Generalization: the text draw a broad conclusion based on a limited sample of population.',
                   'False Cause: the text assume two correlated events must also have a causal relation.',
                   'False Analogy: the text assume two alike things must be alike in other aspects.',
                   'Vagueness: the text use ambiguous words, terms, or phrases.']

if dataset_name == "propaganda":
    fallacy_list = ['Emotional Language', 'Name Calling or Labeling', 'Fear or Prejudice', 'Doubt',
                    'Exaggeration or Minimization', 'Flag-Waving', 'Irrelevant Authority', 'Slogans',
                    'Causal Oversimplification', 'Black-and-White Fallacy', 'Whataboutism', 'Red Herring',
                    'Thought-terminating Cliches', 'Reductio ad hitlerum']
    fallacy_def = ['Emotional Language: the text arouse non-rational emotions.',
                   'Name Calling or Labeling: the text attack a person by assigning a name or label.',
                   'Fear or Prejudice: the text evoke people\'s fear, anxiety, bias, prejudice to persuade them.',
                   'Doubt: the text question the credibility of someone or something.',
                   'Exaggeration or Minimization: the text make things more extreme or less significant.',
                   'Flag-Waving: the text display patriotism or nationalism to promote an action.',
                   'Irrelevant Authority: the text cite an authority but the authority lacks relevant expertise.',
                   'Slogans: the text use a brief and striking phrase to provoke excitement of the audience.',
                   'Causal Oversimplification: the text assume one single cause and ignore other possible causes.',
                   'Black-and-White Fallacy: the text present two alternative options as the only possibilities.',
                   'Whataboutism: the text discredit an opponent\'s position by charging them with hypocrisy.',
                   'Red Herring: the text diverge the attention to irrelevant issues.',
                   'Thought-terminating Cliches: the text use short phrase to discourage critical thinking.',
                   'Reductio ad hitlerum: the text mention the most universally hated figures such as Nazis.']






''' read data '''

def filter_data(split_set):
    original_data = pd.read_csv("./processed_datasets/" + dataset_name + "_" + split_set + ".tsv", sep='\t', header=0)
    text_list = []
    fallacy_label_list = []
    for row_i in range(original_data.shape[0]):
        if original_data['fallacy_label'][row_i] in fallacy_list:
            text_list.append(original_data['text'][row_i])
            fallacy_label_list.append(original_data['fallacy_label'][row_i])

    filtered_data = pd.DataFrame({"fallacy_label": fallacy_label_list, "text": text_list})
    return filtered_data


train = filter_data("train")
dev = filter_data("dev")
test = filter_data("test")

if args.smoke: # [FIX] quick end-to-end check
    train = train[:50].reset_index(drop=True)
    dev = dev[:20].reset_index(drop=True)
    test = test[:20].reset_index(drop=True)




''' logical relation connectives '''

conjuction = ['and', 'as well as', 'as well', 'also', 'separately']
alternative = ['or', 'either', 'instead', 'alternatively', 'else', 'nor', 'neither']
restatement = ['specifically', 'particularly', 'in particular', 'besides', 'additionally', 'in addition', 'moreover',
               'furthermore', 'further', 'plus', 'not only', 'indeed', 'in other words', 'in fact', 'in short',
               'in the end', 'overall', 'in sum', 'in summary', 'in detail', 'in details']
instantiation = ['for example', 'for instance', 'such as', 'including', 'as an example', 'for one thing']

contrast = ['but', 'however', 'yet', 'while', 'unlike', 'rather', 'rather than', 'in comparison', 'by comparison',
            'on the other hand', 'on the contrary', 'contrary to', 'in contrast', 'by contrast', 'still', 'whereas',
            'conversely', 'not', 'no', 'none', 'nothing', 'n\'t']
concession = ['although', 'though', 'despite', 'despite of', 'in spite of', 'regardless', 'regardless of', 'whether',
              'nevertheless', 'nonetheless', 'even if', 'even though', 'even as', 'even when', 'even after',
              'even so', 'no matter']
analogy = ['likewise', 'similarly', 'as if', 'as though', 'just as', 'just like', 'namely']

temporal = ['during', 'before', 'after', 'when', 'as soon as', 'then', 'next', 'until', 'till', 'meanwhile', 'in turn',
            'meantime', 'afterwards', 'afterward', 'simultaneously', 'at the same time', 'beforehand', 'previous',
            'previously', 'earlier', 'later', 'thereafter', 'finally', 'ultimately', 'eventually', 'subsequently']

condition = ['if', 'as long as', 'unless', 'otherwise', 'except', 'whenever', 'whichever', 'provided', 'once',
             'only if', 'only when', 'depend on', 'depends on', 'depending on', 'in case']
causal = ['because', 'cause', 'as a result', 'result in', 'due to', 'therefore', 'hence', 'thus', 'thereby', 'since',
          'now that', 'consequently', 'in consequence', 'in order to', 'so as to', 'so that', 'so', 'as', 'why', 'for',
          'accordingly', 'given', 'turn out', 'turns out']

all_keywords = conjuction + alternative + restatement + instantiation + contrast + concession + analogy + temporal + condition + causal




''' [NEW] relation names, verifier routing and direction (Stage 2) '''

# same 1..10 indices as index_logical_relation in the original code
RELATION_NAMES = [None, "conjunction", "alternative", "restatement", "instantiation", "contrast",
                  "concession", "analogy", "temporal", "condition", "causal"]
RELATION_KEYWORD_LISTS = [None, conjuction, alternative, restatement, instantiation, contrast,
                          concession, analogy, temporal, condition, causal]


def keyword_to_relation(keyword):
    ''' [NEW] map a matched connective to its relation name (same order as the original if-chain) '''
    for relation_i in range(1, 11):
        if keyword in RELATION_KEYWORD_LISTS[relation_i]:
            return RELATION_NAMES[relation_i]
    return None


# Stage 2 table: which verifier checks which relation
RELATION_VERIFIER = {
    "causal": "nli+judge",            # Does the reason justify the conclusion?
    "condition": "nli+judge",         # Does the consequent plausibly follow?
    "instantiation": "judge",         # Does the example support the claim?
    "analogy": "judge",               # Alike in the respect that matters?
    "restatement": "nli_both_ways",   # Does one side just repeat the other? (circularity check only)
    "alternative": "judge",           # Are the options exhaustive?
    "temporal": "rule",               # Flagged when beneath a causal node (Stage 3)
    "contrast": "nli",                # Are the arguments in tension?
    "concession": "nli",              # Are the arguments in tension?
    "conjunction": "none",            # Nothing to verify -> N/A
}

# [FIX] connectives that stay in the tree but are NOT verified (verdict N/A):
# the original lists contain negation words (contrast list) and words that are mostly prepositions ('for', 'as');
# purpose connectives and these condition words are not reason -> conclusion links that entailment can check.
NEGATION_WORDS = ['not', 'no', 'none', 'nothing', 'n\'t']
AMBIGUOUS_CONNECTIVES = ['for', 'as']
PURPOSE_CONNECTIVES = ['in order to', 'so as to', 'so that']
NON_INFERENTIAL = ['turn out', 'turns out', 'unless', 'except', 'otherwise', 'whichever']
SKIP_VERIFICATION = NEGATION_WORDS + AMBIGUOUS_CONNECTIVES + PURPOSE_CONNECTIVES + NON_INFERENTIAL

# Stage 2 direction box: reason = alpha for therefore, so, thus, hence; reason = beta for because, since, due to, given
REASON_IS_ALPHA = ['therefore', 'so', 'thus', 'hence']
REASON_IS_BETA = ['because', 'since', 'due to', 'given']
# remaining causal connectives that are verified, assigned by their usual reading ("X cause Y" -> reason = alpha)
REASON_IS_ALPHA_EXTRA = ['cause', 'as a result', 'result in', 'thereby', 'consequently', 'in consequence',
                         'accordingly', 'why']               # [FIX] 'turn out(s)' moved to NON_INFERENTIAL
REASON_IS_BETA_EXTRA = ['now that']                          # [FIX] 'as', 'for' and purpose connectives are not verified
# For condition, the antecedent (the "reason" side) is beta ("alpha if beta"); [FIX] 'otherwise' is no longer verified.

VERDICTS = ["supported", "unsupported", "circular", "N/A"]
SEVERITY = {"N/A": 0, "supported": 1, "circular": 2, "unsupported": 3}  # Stage 3: weakest link wins; N/A is neutral
SEVERITY_INV = {value: key for key, value in SEVERITY.items()}


def reason_side_of(keyword, relation):
    ''' [NEW] 0 if the reason / antecedent is alpha, 1 if it is beta '''
    if relation == "causal":
        return 1 if keyword in REASON_IS_BETA + REASON_IS_BETA_EXTRA else 0
    if relation == "condition":
        return 1 # [FIX] antecedent is beta for every verified condition connective
    return 0


def is_verifiable(keyword):
    ''' [FIX] False for connectives that stay in the tree but are not verified '''
    return keyword not in SKIP_VERIFICATION


def needs_nli(keyword, relation): # [FIX] also receives the keyword
    ''' [NEW] NLI scores are computed for NLI-routed relations, unless the configuration is judge only or none '''
    if not is_verifiable(keyword):
        return False
    return VERIFIER_MODE in ("nli", "both") and RELATION_VERIFIER[relation] in ("nli+judge", "nli", "nli_both_ways")


def needs_judge(keyword, relation): # [FIX] also receives the keyword
    ''' [NEW] the judge runs for judge-routed relations ("both"), or for every verifiable relation ("judge" only) '''
    if not is_verifiable(keyword):
        return False
    if RELATION_VERIFIER[relation] in ("rule", "none"):
        return False
    if VERIFIER_MODE == "judge":
        return True
    if VERIFIER_MODE == "both":
        return RELATION_VERIFIER[relation] in ("nli+judge", "judge")
    return False




''' construct logical structure tree '''

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')


def recover_text_from_tree(constituency_tree):
    ''' extract the text from the given constituency tree '''

    text = " ".join(constituency_tree.leaves())

    return text


def match_keyword(constituency_tree, previous_matched_keyword_list):
    ''' from top-down and left-right traverse the constituency tree, find the leftmost longest matched keyword '''

    matched_flag = 0
    matched_keyword = ""
    matched_subtree = None

    for subtree in constituency_tree.subtrees():
        text_of_subtree = recover_text_from_tree(subtree)
        if text_of_subtree in all_keywords:
            presented_flag = 0
            for previous_matched_keyword in previous_matched_keyword_list:
                if text_of_subtree in previous_matched_keyword: # the matched keyword is part of or equal to the previous matched keyword
                    presented_flag = 1
                    break

            if presented_flag == 0:
                matched_flag = 1
                matched_keyword = text_of_subtree
                matched_subtree = subtree
                break

    return matched_flag, matched_keyword, matched_subtree


def extract_arguments(matched_keyword, matched_subtree):
    ''' extract the left and right argument of the matched keyword '''

    parent_text = recover_text_from_tree(matched_subtree.parent())

    right_argument = parent_text.split(matched_keyword)[-1]

    if parent_text.split(matched_keyword)[0] != "": # parent = alpha + keyword + betta
        left_argument = parent_text.split(matched_keyword)[0]
    else: # parent = keyword + betta
        if matched_subtree.parent().parent() is not None:
            grandparent_text = recover_text_from_tree(matched_subtree.parent().parent())
            if grandparent_text.split(parent_text)[0] == "":
                if matched_subtree.parent().parent().label()[:5] == "sent_":
                    if matched_subtree.parent().parent().left_sibling() is not None: # exist previous sentence
                        left_argument = recover_text_from_tree(matched_subtree.parent().parent().left_sibling())
                    else:
                        left_argument = ""
                else:
                    left_argument = ""
            else:
                left_argument = grandparent_text.split(parent_text)[0]
        else:
            left_argument = ""

    if len(right_argument) != 0: # delete starting and ending empty space
        if right_argument[0] == " ":
            right_argument = right_argument[1:]
        if right_argument[-1] == " ":
            right_argument = right_argument[:-1]

    if len(left_argument) != 0: # delete starting and ending empty space
        if left_argument[0] == " ":
            left_argument = left_argument[1:]
        if left_argument[-1] == " ":
            left_argument = left_argument[:-1]

    return left_argument, right_argument


def construct_logical_structure_tree(text):
    ''' construct the logical structure tree for the given text '''

    doc = nlp(text.lower())
    constituency_tree_string = "(Root"

    for sent_i in range(len(doc.sentences)):
        sentence_tree = doc.sentences[sent_i].constituency
        sentence_tree.label = "sent_" + str(sent_i)
        constituency_tree_string += " " + str(sentence_tree)

    constituency_tree_string += ")"
    constituency_tree = ParentedTree.fromstring(constituency_tree_string)


    logical_structure_tree = []
    # the logical relations in each sentence are saved from macro to micro perspective

    for sent_i in range(len(constituency_tree)):
        sentence_text = recover_text_from_tree(constituency_tree[sent_i])
        logical_structure_this_sentence = {"sentence_id": sent_i, "sentence_text": sentence_text, "logical_relation": []}

        # initialize
        previous_matched_keyword_list = []
        matched_flag = 1

        # recursively match keyword from the sub constituency_tree that corresponds to the betta argument
        while matched_flag != 0:
            matched_flag, matched_keyword, matched_subtree = match_keyword(constituency_tree[sent_i], previous_matched_keyword_list)
            if matched_flag == 1:
                previous_matched_keyword_list.append(matched_keyword)
                left_argument, right_argument = extract_arguments(matched_keyword, matched_subtree)
                logical_structure_this_sentence["logical_relation"].append({"logical_keyword": matched_keyword, "left_argument": left_argument, "right_argument": right_argument})

        logical_structure_tree.append(logical_structure_this_sentence)

    return logical_structure_tree





''' [NEW] Stage 1 cache and tree structure '''

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./saved_models/", exist_ok=True)


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def build_or_load_trees(dataframe, split_set):
    ''' [NEW] Stage 1 is unchanged; its output is computed once and cached, instead of re-parsing every epoch '''
    path = os.path.join(CACHE_DIR, dataset_name + "_" + split_set + ("_smoke" if args.smoke else "") + "_trees.json") # [FIX] smoke runs never overwrite the full cache
    trees = load_json(path, None)
    if trees is None or len(trees) != dataframe.shape[0]:
        trees = []
        for text in tqdm(dataframe['text'], desc="Stage 1 trees (" + split_set + ")"):
            trees.append(construct_logical_structure_tree(text))
        save_json(trees, path)
    return trees


def span_inside(container, keyword, left_argument, right_argument):
    ''' [NEW] True if the relation (keyword + its argument) lies inside the container argument text '''
    container_padded = " " + container + " "
    if (" " + keyword + " ") not in container_padded:
        return False
    inner = right_argument if right_argument != "" else left_argument
    if inner == "":
        return False
    return (" " + inner + " ") in container_padded


def derive_tree_structure(relations):
    ''' [NEW] recover parent / side (0 = alpha, 1 = beta) of every relation node from the Stage 1 output.
    The parent is the relation whose alpha or beta contains this relation, choosing the smallest such argument.
    A parent must span more text than its child, so the result is always a tree (no cycles). '''

    num_relations = len(relations)
    span_length = [len(r["left_argument"]) + len(r["right_argument"]) for r in relations]
    structure = []

    for j in range(num_relations):
        best_parent, best_side, best_length = -1, -1, float("inf")
        for i in range(num_relations):
            if i == j or span_length[i] <= span_length[j]:
                continue
            for side, argument in ((0, relations[i]["left_argument"]), (1, relations[i]["right_argument"])):
                if span_inside(argument, relations[j]["logical_keyword"], relations[j]["left_argument"], relations[j]["right_argument"]):
                    if len(argument) < best_length:
                        best_parent, best_side, best_length = i, side, len(argument)
        structure.append((best_parent, best_side))

    return structure


def node_depths(parents):
    ''' [NEW] depth of every node given its parent index (-1 = root) '''
    depth = [None] * len(parents)

    def get_depth(i):
        if depth[i] is None:
            depth[i] = 0 if parents[i] == -1 else get_depth(parents[i]) + 1
        return depth[i]

    for i in range(len(parents)):
        get_depth(i)
    return depth


def nli_key(left_argument, right_argument):
    return left_argument + "||" + right_argument


def node_key(sentence_text, node):
    return "||".join([sentence_text, node["logical_keyword"], node["left_argument"], node["right_argument"]])


def clean_span(text):
    return text.strip(" ,.;:")




''' [NEW] Stage 2: semantic relation verification '''

JUDGE_SYSTEM_PROMPT = ("You check whether one logical relation in a piece of text holds. "
                       "Judge only the relation you are asked about, using the sentence as context.")


def build_judge_messages(relation, keyword, left_argument, right_argument, reason_side, sentence_text):
    ''' [NEW] relation question only, no fallacy names (Stage 2, LLM-as-a-judge) '''

    alpha, beta = clean_span(left_argument), clean_span(right_argument)

    if relation == "causal":
        reason, conclusion = (alpha, beta) if reason_side == 0 else (beta, alpha)
        roles = "Reason: " + reason + "\nConclusion: " + conclusion
        question = "Does the reason justify the conclusion?"
    elif relation == "condition":
        antecedent, consequent = (alpha, beta) if reason_side == 0 else (beta, alpha)
        roles = "Condition: " + antecedent + "\nConsequent: " + consequent
        question = "Does the consequent plausibly follow from the condition?"
    elif relation == "instantiation":
        roles = "Claim: " + alpha + "\nExample: " + beta
        question = "Does the example support the claim?"
    elif relation == "analogy":
        roles = "First: " + alpha + "\nSecond: " + beta
        question = "Are the two alike in the respect that matters for the argument?"
    elif relation == "alternative":
        roles = "Option 1: " + alpha + "\nOption 2: " + beta
        question = "Are these options exhaustive, meaning there is no other realistic option?"
    elif relation == "restatement":   # only used in judge-only mode
        roles = "Statement 1: " + alpha + "\nStatement 2: " + beta
        question = "Does one statement just repeat the other without adding support?"
    else:                             # contrast, concession: only used in judge-only mode
        roles = "Statement 1: " + alpha + "\nStatement 2: " + beta
        question = "Are the two statements in tension with each other?"

    user_prompt = ("Sentence (context): " + sentence_text + "\n"
                   "Connective: \"" + keyword + "\" (" + relation + " relation)\n"
                   + roles + "\n"
                   "Question: " + question + "\n"
                   "Answer Yes or No, then give a one-sentence rationale about this relation only.")

    return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]


def run_nli(nli_pairs):
    ''' [NEW] NLI verifier: DeBERTa-v3-large, forward (alpha -> beta) and backward (beta -> alpha),
    returns P(entail, neutral, contradict) for both directions; cached offline.
    The premise / hypothesis are the two arguments only: the sentence contains the conclusion itself,
    so giving it as context would make NLI predict entailment trivially. '''

    cache_path = os.path.join(CACHE_DIR, dataset_name + "_nli_cache_" + NLI_MODEL_NAME.replace("/", "_") + ".json")
    cache = load_json(cache_path, {})
    todo = [key for key in nli_pairs if key not in cache]
    if len(todo) == 0:
        return cache

    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(device)
    nli_model.eval()
    label2id = {label.lower(): index for index, label in nli_model.config.id2label.items()}
    label_order = [label2id["entailment"], label2id["neutral"], label2id["contradiction"]]

    for start in tqdm(range(0, len(todo), NLI_BATCH_SIZE), desc="Stage 2 (NLI verifier)"):
        keys = todo[start:start + NLI_BATCH_SIZE]
        alphas = [clean_span(nli_pairs[key][0]) for key in keys]
        betas = [clean_span(nli_pairs[key][1]) for key in keys]
        encoded = nli_tokenizer(alphas + betas, betas + alphas, truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(nli_model(**encoded).logits.float(), dim=-1)[:, label_order]
        for b, key in enumerate(keys):
            cache[key] = {"fwd": probs[b].tolist(), "bwd": probs[b + len(keys)].tolist()}

    save_json(cache, cache_path)
    del nli_model
    gc.collect()
    torch.cuda.empty_cache()
    return cache


def run_judge(judge_queries):
    ''' [NEW] LLM-as-a-judge: open instruct model, T = 0 (greedy), outputs verdict, confidence, rationale.
    Confidence = P(Yes) or P(No) of the first answer token, renormalised over {Yes, No}. Runs once, cached. '''

    cache_path = os.path.join(CACHE_DIR, dataset_name + "_judge_cache_" + JUDGE_MODEL_NAME.replace("/", "_") + ".json")
    cache = load_json(cache_path, {})
    todo = [key for key in judge_queries if key not in cache]
    if len(todo) == 0:
        return cache

    judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
    judge_model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
    judge_model.eval()
    pad_id = judge_tokenizer.pad_token_id if judge_tokenizer.pad_token_id is not None else judge_tokenizer.eos_token_id
    yes_ids = sorted({judge_tokenizer.encode(w, add_special_tokens=False)[0] for w in [" Yes", "Yes", " yes", "yes"]})
    no_ids = sorted({judge_tokenizer.encode(w, add_special_tokens=False)[0] for w in [" No", "No", " no", "no"]})

    for count, key in enumerate(tqdm(todo, desc="Stage 2 (LLM-as-a-judge)")):
        prompt = judge_tokenizer.apply_chat_template(judge_queries[key], tokenize=False, add_generation_prompt=True) + "Answer:"
        encoded = judge_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            next_token_probs = torch.softmax(judge_model(**encoded).logits[0, -1, :].float(), dim=-1)
            p_yes = next_token_probs[yes_ids].sum().item()
            p_no = next_token_probs[no_ids].sum().item()
            answer = "yes" if p_yes >= p_no else "no"
            confidence = max(p_yes, p_no) / (p_yes + p_no + 1e-12)

            # rationale, generated greedily after the chosen answer so it is consistent with the verdict
            rationale_prompt = prompt + (" Yes." if answer == "yes" else " No.") + " Rationale:"
            encoded_rationale = judge_tokenizer(rationale_prompt, return_tensors="pt", add_special_tokens=False).to(device)
            generated = judge_model.generate(**encoded_rationale, max_new_tokens=JUDGE_MAX_NEW_TOKENS, do_sample=False,
                                             temperature=None, top_p=None, top_k=None, pad_token_id=pad_id)

        rationale = judge_tokenizer.decode(generated[0][encoded_rationale["input_ids"].shape[1]:], skip_special_tokens=True)
        rationale = rationale.strip().split("\n")[0].strip()
        cache[key] = {"answer": answer, "conf": confidence, "rationale": rationale}

        if (count + 1) % 200 == 0:
            save_json(cache, cache_path)

    save_json(cache, cache_path)
    del judge_model
    gc.collect()
    torch.cuda.empty_cache()
    return cache


def nli_decide(relation, nli_fwd, nli_bwd, reason_side):
    ''' [NEW] turn NLI scores into a verdict. Index 0 = entail, 1 = neutral, 2 = contradict.
    circular: entailment both ways (checked for causal, condition, restatement). '''

    entail_fwd, entail_bwd = nli_fwd[0], nli_bwd[0]

    # [FIX] circularity needs strong entailment in BOTH directions (separate, higher threshold)
    if relation in ("causal", "condition", "restatement") and entail_fwd >= NLI_CIRCULAR_THRESHOLD and entail_bwd >= NLI_CIRCULAR_THRESHOLD:
        return "circular", min(entail_fwd, entail_bwd), "the two sides entail each other (NLI)"

    if relation in ("causal", "condition"):
        p_entail = entail_fwd if reason_side == 0 else entail_bwd   # reason -> conclusion, using the direction rule
        if p_entail >= NLI_THRESHOLD:
            return "supported", p_entail, "the reason entails the conclusion (NLI)"
        return "unsupported", 1.0 - p_entail, "the reason does not entail the conclusion (NLI)"

    if relation == "restatement":   # [FIX] restatement is a circularity check only: circular, otherwise N/A
        return "N/A", 0.0, ""

    # contrast, concession: in tension unless one side entails the other
    p_entail = max(entail_fwd, entail_bwd)
    if p_entail >= NLI_THRESHOLD:
        return "unsupported", p_entail, "the two sides agree, so there is no tension (NLI)"
    return "supported", 1.0 - p_entail, "the two sides are in tension (NLI)"


def decide_own_verdict(relation, node, nli_entry, judge_entry):
    ''' [NEW] own(n): verdict of one connective node from its relation-specific check '''

    if not is_verifiable(node["logical_keyword"]):   # [FIX] negation, prepositions, purpose, non-inferential connectives
        return "N/A", 0.0, ""
    if relation == "conjunction":            # nothing to verify
        return "N/A", 0.0, ""
    if relation == "temporal":               # rule-based; flagged in Stage 3 when beneath a causal node
        return "N/A", 0.0, ""
    if node["left_argument"] == "" or node["right_argument"] == "":
        return "N/A", 0.0, ""

    nli_verdict = None
    if nli_entry is not None:
        nli_verdict, nli_conf, nli_rationale = nli_decide(relation, nli_entry["fwd"], nli_entry["bwd"], node["reason_side"])

    judge_verdict = None
    if judge_entry is not None:
        answered_yes = judge_entry["answer"] == "yes"
        if relation == "restatement":
            judge_verdict = "circular" if answered_yes else "N/A"   # [FIX] circularity check only
        else:
            judge_verdict = "supported" if answered_yes else "unsupported"
        judge_conf, judge_rationale = judge_entry["conf"], judge_entry["rationale"]

    # Combined configuration ("both"): circularity (entailment both ways) comes from NLI; otherwise the judge's
    # verdict decides, and the NLI scores still enter the verification vector v as features.
    # (This is a design choice of the combined configuration and is stated as such in the paper.)
    if nli_verdict == "circular":
        return nli_verdict, nli_conf, (judge_rationale if judge_verdict is not None and judge_rationale else nli_rationale)
    if judge_verdict is not None and judge_verdict != "N/A":
        return judge_verdict, judge_conf, judge_rationale
    if nli_verdict is not None and nli_verdict != "N/A":
        return nli_verdict, nli_conf, nli_rationale
    return "N/A", 0.0, ""   # e.g. judge-routed relation in the NLI-only configuration




''' [NEW] Stage 3: verdict aggregation over the tree '''

def aggregate_sentence(relations):
    ''' effective(n) = own(n) AND effective(child relations), bottom-up; a conclusion is only as strong as its weakest link.
    Temporal node beneath a causal node raises a post hoc flag. '''

    num_relations = len(relations)
    parents = [r["parent"] for r in relations]
    depth = node_depths(parents)
    children = [[] for _ in range(num_relations)]
    for j in range(num_relations):
        if parents[j] != -1:
            children[parents[j]].append(j)

    for i in sorted(range(num_relations), key=lambda x: -depth[x]):   # bottom-up
        severity = SEVERITY[relations[i]["verdict"]]
        for child in children[i]:
            severity = max(severity, SEVERITY[relations[child]["effective"]])
        relations[i]["effective"] = SEVERITY_INV[severity]
        relations[i]["depth"] = depth[i]

    for i in range(num_relations):
        relations[i]["flag"] = 0
        relations[i]["posthoc_ancestor"] = -1
        relations[i]["posthoc_on_reason_side"] = False
        if relations[i]["relation"] != "temporal" or VERIFIER_MODE == "none":   # [FIX] the control gets no flags
            continue
        child, ancestor = i, parents[i]
        while ancestor != -1:
            if relations[ancestor]["relation"] == "causal" and is_verifiable(relations[ancestor]["logical_keyword"]): # [FIX]
                relations[i]["flag"] = 1
                relations[i]["posthoc_ancestor"] = ancestor
                relations[i]["posthoc_on_reason_side"] = relations[child]["side"] == relations[ancestor]["reason_side"]
                break
            child, ancestor = ancestor, parents[ancestor]


def build_tree_outputs(sentences):
    ''' tree-level summary (Stage 3), natural-language tree summary (Stage 4 hard prompt),
    and the explanation assembled from failed relations + judge rationales (Stage 5) '''

    nodes = [(s_i, n) for s_i, sentence in enumerate(sentences) for n in sentence["logical_relation"]]
    if len(nodes) == 0:
        tree_level = {"root_effective_verdict": "N/A", "failed_relations": {}, "flags": []}
        return "No logical connectives were found in the Text.", tree_level, "No logical connectives were found, so no relation could be verified."

    roots = [n for _, n in nodes if n["parent"] == -1]
    failed = sorted([(s_i, n) for s_i, n in nodes if n["verdict"] in ("unsupported", "circular")], key=lambda x: -x[1]["depth"])
    flagged = [(s_i, n) for s_i, n in nodes if n["flag"] == 1]

    failed_counts = {}
    for _, n in failed:
        failed_counts[n["relation"]] = failed_counts.get(n["relation"], 0) + 1
    tree_level = {"root_effective_verdict": SEVERITY_INV[max(SEVERITY[n["effective"]] for n in roots)],
                  "failed_relations": failed_counts,
                  "flags": ["temporal beneath causal (post hoc pattern)"] if flagged else []}

    # natural-language tree summary for the hard prompt
    summary = []
    for n in roots:
        if n["effective"] != "N/A" or is_verifiable(n["logical_keyword"]): # [FIX] do not describe skipped connectives as claims
            summary.append("The root " + n["relation"] + " claim (" + n["logical_keyword"] + ") is " + (n["effective"] if n["effective"] != "N/A" else "not checked") + ".")
    for s_i, n in flagged:
        ancestor = sentences[s_i]["logical_relation"][n["posthoc_ancestor"]]
        if n["posthoc_on_reason_side"]:
            summary.append("The reason for the causal claim (" + ancestor["logical_keyword"] + ") is only a temporal sequence (" + n["logical_keyword"] + ").")
        else:
            summary.append("A temporal sequence (" + n["logical_keyword"] + ") appears beneath the causal claim (" + ancestor["logical_keyword"] + ").")
    for _, n in failed:
        if n["parent"] != -1:
            summary.append("The inner " + n["relation"] + " link (" + n["logical_keyword"] + ") is " + n["verdict"] + ".")
    if len(summary) == 0: # [FIX]
        summary.append("No relation in the Text could be verified.")

    # explanation assembled from failed relations + judge rationales (not generated by the classifier)
    explanation = []
    for s_i, n in flagged:
        if n["posthoc_on_reason_side"]:
            ancestor = sentences[s_i]["logical_relation"][n["posthoc_ancestor"]]
            explanation.append("The argument infers causation from temporal order (" + n["logical_keyword"] + " … " + ancestor["logical_keyword"] + ").")
    for _, n in failed:
        alpha, beta = clean_span(n["left_argument"]), clean_span(n["right_argument"])
        if n["relation"] in ("causal", "condition") and n["reason_side"] == 1:
            alpha, beta = beta, alpha
        if n["verdict"] == "circular":
            sentence = "The " + n["relation"] + " relation (" + n["logical_keyword"] + ") is circular: \"" + alpha + "\" and \"" + beta + "\" restate each other"
        else:
            sentence = "The " + n["relation"] + " link from \"" + alpha + "\" to \"" + beta + "\" is " + n["verdict"]
        if n["rationale"]:
            sentence += ": " + n["rationale"].rstrip(".")
        explanation.append(sentence + ".")
    if len(explanation) == 0:
        explanation.append("Every verified relation is supported; no relation failed verification.")

    return " ".join(summary), tree_level, " ".join(explanation)


def verify_tree(tree, nli_cache, judge_cache):
    ''' [FIX] Stage 2 only (own verdicts + structure), split from Stage 3 so verdicts can be shuffled in between.
    The Stage 1 tree itself is not modified. '''

    sentences = copy.deepcopy(tree)
    for sentence in sentences:
        relations = sentence["logical_relation"]
        structure = derive_tree_structure(relations)

        for i, node in enumerate(relations):
            relation = keyword_to_relation(node["logical_keyword"])
            node["relation"] = relation
            node["parent"], node["side"] = structure[i]
            node["reason_side"] = reason_side_of(node["logical_keyword"], relation)

            has_arguments = node["left_argument"] != "" and node["right_argument"] != ""
            nli_entry = nli_cache.get(nli_key(node["left_argument"], node["right_argument"])) if (has_arguments and needs_nli(node["logical_keyword"], relation)) else None
            judge_entry = judge_cache.get(node_key(sentence["sentence_text"], node)) if (has_arguments and needs_judge(node["logical_keyword"], relation)) else None

            node["nli_fwd"] = nli_entry["fwd"] if nli_entry is not None else [0.0, 0.0, 0.0]
            node["nli_bwd"] = nli_entry["bwd"] if nli_entry is not None else [0.0, 0.0, 0.0]
            node["verdict"], node["conf"], node["rationale"] = decide_own_verdict(relation, node, nli_entry, judge_entry)

    return sentences


def shuffle_verdicts(split_sentences, rng):
    ''' [FIX] ablation: randomly reassign the Stage 2 outputs (verdict, confidence, rationale, NLI scores) among all
    checked relations of a split. The verdict distribution is unchanged, but its link to the content is broken. '''

    checked = [node for sentences in split_sentences for sentence in sentences for node in sentence["logical_relation"]
               if node["verdict"] != "N/A"]
    payloads = [(n["verdict"], n["conf"], n["rationale"], n["nli_fwd"], n["nli_bwd"]) for n in checked]
    rng.shuffle(payloads)
    for node, payload in zip(checked, payloads):
        node["verdict"], node["conf"], node["rationale"], node["nli_fwd"], node["nli_bwd"] = payload


def aggregate_tree(sentences):
    ''' [FIX] Stage 3 (aggregation) + outputs for one text '''

    for sentence in sentences:
        aggregate_sentence(sentence["logical_relation"])
    tree_summary, tree_level, explanation = build_tree_outputs(sentences)
    return {"sentences": sentences, "tree_summary": tree_summary, "tree_level": tree_level, "explanation": explanation}


def format_verdict(node):
    ''' [NEW] verdict column of the verified textualized tree, e.g. "unsupported (0.86)" or "N/A + flag" '''
    text = node["verdict"] if node["verdict"] == "N/A" else node["verdict"] + " ({:.2f})".format(node["conf"])
    if node["flag"] == 1:
        text += " + flag"
    return text


def prepare_verified_trees():
    ''' [NEW] run Stages 1-3 for all splits before Llama-2 is loaded; NLI and judge models are freed afterwards '''

    split_frames = {"train": train, "dev": dev, "test": test}
    split_trees = {split: build_or_load_trees(frame, split) for split, frame in split_frames.items()}

    nli_pairs, judge_queries = {}, {}
    for trees in split_trees.values():
        for tree in trees:
            for sentence in tree:
                for node in sentence["logical_relation"]:
                    if node["left_argument"] == "" or node["right_argument"] == "":
                        continue
                    relation = keyword_to_relation(node["logical_keyword"])
                    if needs_nli(node["logical_keyword"], relation):   # [FIX] skipped connectives are never sent to a verifier
                        nli_pairs[nli_key(node["left_argument"], node["right_argument"])] = (node["left_argument"], node["right_argument"])
                    if needs_judge(node["logical_keyword"], relation): # [FIX]
                        judge_queries[node_key(sentence["sentence_text"], node)] = build_judge_messages(
                            relation, node["logical_keyword"], node["left_argument"], node["right_argument"],
                            reason_side_of(node["logical_keyword"], relation), sentence["sentence_text"])

    print("Stage 2: {} NLI pairs, {} judge queries".format(len(nli_pairs), len(judge_queries))) # [FIX]
    nli_cache = run_nli(nli_pairs) if len(nli_pairs) != 0 else {}
    judge_cache = run_judge(judge_queries) if len(judge_queries) != 0 else {}

    # [FIX] Stage 2 for every tree, optional shuffling ablation, then Stage 3
    rng = random.Random(seed_val)
    verified = {}
    for split, trees in split_trees.items():
        split_sentences = [verify_tree(tree, nli_cache, judge_cache) for tree in trees]
        if SHUFFLE_VERDICTS:
            shuffle_verdicts(split_sentences, rng)
        verified[split] = [aggregate_tree(sentences) for sentences in split_sentences]
    return verified


verified_trees = prepare_verified_trees()





''' custom dataset '''

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tree_tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")


class custom_dataset(Dataset):
    def __init__(self, dataframe, verified_trees_split): # [CHANGED] also receives the verified trees of this split
        self.dataframe = dataframe
        self.verified_trees_split = verified_trees_split # [NEW]

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):

        text = self.dataframe['text'][idx]
        fallacy_label = self.dataframe['fallacy_label'][idx]

        # [CHANGED] the Stage 1 tree (same output as construct_logical_structure_tree(text)) now carries Stage 2-3 results
        verified_tree = self.verified_trees_split[idx]
        logical_structure_tree = verified_tree["sentences"]


        ''' pre_prompt, fallacy_list, fallacy_def '''

        instruction_prompt = "<s>The task is to classify the fallacy type of the Text. Choose one answer from these fallacy types: "
        instruction_prompt += ", ".join(fallacy_list) + ". "
        instruction_prompt += "The definitions of each fallacy type are as follows. "
        instruction_prompt += " ".join(fallacy_def) + "\n"


        ''' textualized tree '''

        # [CHANGED] Stage 4 hard prompt: LST triplet table plus a verdict column (rows bottom-up, as in LST), then the tree summary
        # [FIX] without verdict text (ablation / control), the table keeps the relation names but has no verdict column or summary
        if USE_VERDICT_TEXT:
            textualized_tree = "The logical relations in the Text, and whether each is supported by its arguments, are presented in this table: argument 1\tlogical relation\targument 2\tverdict\n"
        else:
            textualized_tree = "The logical relations in the Text are presented in this table: argument 1\tlogical relation\targument 2\n"

        for sent_i in range(len(logical_structure_tree)):
            for logic_relation_i in range(len(logical_structure_tree[sent_i]["logical_relation"]) - 1, -1, -1): # bottom-up textualize the tree
                node = logical_structure_tree[sent_i]["logical_relation"][logic_relation_i] # [NEW]
                logical_keyword = node["logical_keyword"]
                left_argument = node["left_argument"]
                right_argument = node["right_argument"]
                textualized_tree += left_argument + "\t" + logical_keyword + " (" + node["relation"] + ")\t" + right_argument # [CHANGED]
                textualized_tree += ("\t" + format_verdict(node) + "\n") if USE_VERDICT_TEXT else "\n"               # [FIX]

        tree_summary = ("Tree summary: " + verified_tree["tree_summary"] + "\n") if USE_VERDICT_TEXT else "" # [NEW] [FIX]


        ''' source_input_ids, llama_soft_token '''

        source_input_ids = llama_tokenizer(instruction_prompt, add_special_tokens=False).input_ids
        source_input_ids += llama_tokenizer(textualized_tree, add_special_tokens=False).input_ids
        source_input_ids += llama_tokenizer(tree_summary, add_special_tokens=False).input_ids # [NEW]
        source_input_ids += llama_tokenizer("Please classify the fallacy type of the Text. Text: ", add_special_tokens=False).input_ids
        llama_soft_token = [len(source_input_ids)] # the index in source_input_ids where to add soft token
        source_input_ids += llama_tokenizer(text, add_special_tokens=False).input_ids
        source_input_ids += llama_tokenizer(" Answer:", add_special_tokens=False).input_ids if text[-1] == "." or text[-1] == "?" or text[-1] == "!" or text[-1] == "\n" else llama_tokenizer(". Answer:", add_special_tokens=False).input_ids

        if len(source_input_ids) > max_source_length:
            num_delete_token = len(source_input_ids) - max_source_length
            source_input_ids = llama_tokenizer(instruction_prompt, add_special_tokens=False).input_ids
            source_input_ids += llama_tokenizer(textualized_tree, add_special_tokens=False).input_ids[:-num_delete_token]
            source_input_ids += llama_tokenizer(tree_summary, add_special_tokens=False).input_ids # [NEW] the summary is kept; the table is truncated
            source_input_ids += llama_tokenizer("Please classify the fallacy type of the Text. Text: ", add_special_tokens=False).input_ids
            llama_soft_token = [len(source_input_ids)]
            source_input_ids += llama_tokenizer(text, add_special_tokens=False).input_ids
            source_input_ids += llama_tokenizer(" Answer:", add_special_tokens=False).input_ids if text[-1] == "." or text[-1] == "?" or text[-1] == "!" or text[-1] == "\n" else llama_tokenizer(". Answer:", add_special_tokens=False).input_ids

        source_input_ids = torch.tensor(source_input_ids)
        source_input_ids = source_input_ids.view(1, source_input_ids.shape[0])
        llama_soft_token = torch.tensor(llama_soft_token)


        ''' target_input_ids '''

        target_input_ids = llama_tokenizer(fallacy_label + "</s>", add_special_tokens=False).input_ids
        target_input_ids = torch.tensor(target_input_ids)
        target_input_ids = target_input_ids.view(1, target_input_ids.shape[0])


        ''' tree_input_ids, tree_attention_mask '''

        # used to derive logical structure tree embedding

        logical_keyword_list = []
        left_argument_list = []
        right_argument_list = []
        sentence_text_list = []

        num_logical_relations_each_sent = [] # the number of logical relations in each sentence
        index_logical_relation = [] # the index of logical relation that the logical keyword belongs to

        # [NEW] tree structure and Stage 2-3 features per relation node, in the same bottom-up order
        parent_index = []   # parent position within the sentence (bottom-up order), -1 for a root
        parent_side = []    # 0 = the node sits in its parent's alpha, 1 = in beta, -1 for a root
        nli_scores = []     # [P_NLI(alpha->beta) ; P_NLI(beta->alpha)]
        verdict_ids = []    # own verdict, index into VERDICTS
        verdict_conf = []   # confidence of the verdict
        flags = []          # post hoc flag

        for sent_i in range(len(logical_structure_tree)):
            num_logical_relations_each_sent.append(len(logical_structure_tree[sent_i]["logical_relation"]))
            sentence_text_list.append(logical_structure_tree[sent_i]["sentence_text"])
            num_relations_this_sent = len(logical_structure_tree[sent_i]["logical_relation"]) # [NEW]

            for logic_relation_i in range(len(logical_structure_tree[sent_i]["logical_relation"]) - 1, -1, -1):
                logical_keyword = logical_structure_tree[sent_i]["logical_relation"][logic_relation_i]["logical_keyword"]
                left_argument = logical_structure_tree[sent_i]["logical_relation"][logic_relation_i]["left_argument"]
                right_argument = logical_structure_tree[sent_i]["logical_relation"][logic_relation_i]["right_argument"]

                logical_keyword_list.append(" " + logical_keyword) # add space before, for roberta tokenizer
                left_argument_list.append(" " + left_argument)
                right_argument_list.append(" " + right_argument)

                if logical_keyword in conjuction:
                    index_logical_relation.append(1)
                if logical_keyword in alternative:
                    index_logical_relation.append(2)
                if logical_keyword in restatement:
                    index_logical_relation.append(3)
                if logical_keyword in instantiation:
                    index_logical_relation.append(4)
                if logical_keyword in contrast:
                    index_logical_relation.append(5)
                if logical_keyword in concession:
                    index_logical_relation.append(6)
                if logical_keyword in analogy:
                    index_logical_relation.append(7)
                if logical_keyword in temporal:
                    index_logical_relation.append(8)
                if logical_keyword in condition:
                    index_logical_relation.append(9)
                if logical_keyword in causal:
                    index_logical_relation.append(10)

                # [NEW] remap the parent index to the reversed (bottom-up) order used here
                node = logical_structure_tree[sent_i]["logical_relation"][logic_relation_i]
                parent_index.append(-1 if node["parent"] == -1 else num_relations_this_sent - 1 - node["parent"])
                parent_side.append(node["side"])
                nli_scores.append(node["nli_fwd"] + node["nli_bwd"])
                verdict_ids.append(VERDICTS.index(node["verdict"]))
                verdict_conf.append(float(node["conf"]))
                flags.append(float(node["flag"]))

        if len(sentence_text_list) != 0:
            sentence_text_input_ids = tree_tokenizer(sentence_text_list, add_special_tokens=False, padding='longest')["input_ids"]
            sentence_text_attention_mask = tree_tokenizer(sentence_text_list, add_special_tokens=False, padding='longest')["attention_mask"]

            sentence_text_input_ids = torch.tensor(sentence_text_input_ids)
            sentence_text_attention_mask = torch.tensor(sentence_text_attention_mask)

            num_logical_relations_each_sent = torch.tensor(num_logical_relations_each_sent)
        else:
            sentence_text_input_ids = torch.tensor([])
            sentence_text_attention_mask = torch.tensor([])
            num_logical_relations_each_sent = torch.tensor([])


        if len(logical_keyword_list) != 0:
            logical_keyword_input_ids = tree_tokenizer(logical_keyword_list, add_special_tokens=False, padding='longest')["input_ids"]
            logical_keyword_attention_mask = tree_tokenizer(logical_keyword_list, add_special_tokens=False, padding='longest')["attention_mask"]
            left_argument_input_ids = tree_tokenizer(left_argument_list, add_special_tokens=False, padding='longest')["input_ids"]
            left_argument_attention_mask = tree_tokenizer(left_argument_list, add_special_tokens=False, padding='longest')["attention_mask"]
            right_argument_input_ids = tree_tokenizer(right_argument_list, add_special_tokens=False, padding='longest')["input_ids"]
            right_argument_attention_mask = tree_tokenizer(right_argument_list, add_special_tokens=False, padding='longest')["attention_mask"]

            logical_keyword_input_ids = torch.tensor(logical_keyword_input_ids)
            logical_keyword_attention_mask = torch.tensor(logical_keyword_attention_mask)
            left_argument_input_ids = torch.tensor(left_argument_input_ids)
            left_argument_attention_mask = torch.tensor(left_argument_attention_mask)
            right_argument_input_ids = torch.tensor(right_argument_input_ids)
            right_argument_attention_mask = torch.tensor(right_argument_attention_mask)

            index_logical_relation = torch.tensor(index_logical_relation)

            # [NEW]
            parent_index = torch.tensor(parent_index, dtype=torch.long)
            parent_side = torch.tensor(parent_side, dtype=torch.long)
            nli_scores = torch.tensor(nli_scores, dtype=torch.float32)
            verdict_ids = torch.tensor(verdict_ids, dtype=torch.long)
            verdict_conf = torch.tensor(verdict_conf, dtype=torch.float32)
            flags = torch.tensor(flags, dtype=torch.float32)
        else:
            logical_keyword_input_ids = torch.tensor([])
            logical_keyword_attention_mask = torch.tensor([])
            left_argument_input_ids = torch.tensor([])
            left_argument_attention_mask = torch.tensor([])
            right_argument_input_ids = torch.tensor([])
            right_argument_attention_mask = torch.tensor([])
            index_logical_relation = torch.tensor([])

            # [NEW]
            parent_index = torch.tensor([])
            parent_side = torch.tensor([])
            nli_scores = torch.tensor([])
            verdict_ids = torch.tensor([])
            verdict_conf = torch.tensor([])
            flags = torch.tensor([])



        dict = {"source_input_ids": source_input_ids, "target_input_ids": target_input_ids, "llama_soft_token": llama_soft_token,
                "logical_keyword_input_ids": logical_keyword_input_ids, "logical_keyword_attention_mask": logical_keyword_attention_mask,
                "left_argument_input_ids": left_argument_input_ids, "left_argument_attention_mask": left_argument_attention_mask,
                "right_argument_input_ids": right_argument_input_ids, "right_argument_attention_mask": right_argument_attention_mask,
                "sentence_text_input_ids": sentence_text_input_ids, "sentence_text_attention_mask": sentence_text_attention_mask,
                "num_logical_relations_each_sent": num_logical_relations_each_sent, "index_logical_relation": index_logical_relation,
                # [NEW] Stage 2-4 inputs, and Stage 5 outputs used at test time
                "parent_index": parent_index, "parent_side": parent_side, "nli_scores": nli_scores,
                "verdict_ids": verdict_ids, "verdict_conf": verdict_conf, "flags": flags,
                "text": text, "tree_summary": verified_tree["tree_summary"], "explanation": verified_tree["explanation"],
                "tree_level_json": json.dumps(verified_tree["tree_level"])}

        return dict





''' model '''

class Tree_Embedding(nn.Module):

    def __init__(self):
        super(Tree_Embedding, self).__init__()

        self.roberta = RobertaModel.from_pretrained("FacebookAI/roberta-base", output_hidden_states=True, )

        self.projection_layer_1 = nn.Linear(768, 2048, bias=True)
        nn.init.xavier_uniform_(self.projection_layer_1.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.projection_layer_1.bias)

        self.projection_layer_2 = nn.Linear(2048, 4096, bias=True)
        nn.init.xavier_uniform_(self.projection_layer_2.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.projection_layer_2.bias)

        self.W_text_tree = nn.Linear(768 * 2, 768, bias=True)
        nn.init.xavier_uniform_(self.W_text_tree.weight)
        nn.init.zeros_(self.W_text_tree.bias)

        # [CHANGED] relation-specific encoders are now conditioned on the verdict: e = W^r(e_l + e_c + e_r + v) + b^r
        # (in the gated variant these are W^r_sup)
        self.W_conjuction = nn.Linear(768 * 3 + V_DIM, 768, bias=True)
        nn.init.xavier_uniform_(self.W_conjuction.weight)
        nn.init.zeros_(self.W_conjuction.bias)

        self.W_alternative = nn.Linear(768 * 3 + V_DIM, 768, bias=True)
        nn.init.xavier_uniform_(self.W_alternative.weight)
        nn.init.zeros_(self.W_alternative.bias)

        self.W_restatement = nn.Linear(768 * 3 + V_DIM, 768, bias=True)
        nn.init.xavier_uniform_(self.W_restatement.weight)
        nn.init.zeros_(self.W_restatement.bias)

        self.W_instantiation = nn.Linear(768 * 3 + V_DIM, 768, bias=True)
        nn.init.xavier_uniform_(self.W_instantiation.weight)
        nn.init.zeros_(self.W_instantiation.bias)

        self.W_contrast = nn.Linear(768 * 3 + V_DIM, 768, bias=True)
        nn.init.xavier_uniform_(self.W_contrast.weight)
        nn.init.zeros_(self.W_contrast.bias)

        self.W_concession = nn.Linear(768 * 3 + V_DIM, 768, bias=True)
        nn.init.xavier_uniform_(self.W_concession.weight)
        nn.init.zeros_(self.W_concession.bias)

        self.W_analogy = nn.Linear(768 * 3 + V_DIM, 768, bias=True)
        nn.init.xavier_uniform_(self.W_analogy.weight)
        nn.init.zeros_(self.W_analogy.bias)

        self.W_temporal = nn.Linear(768 * 3 + V_DIM, 768, bias=True)
        nn.init.xavier_uniform_(self.W_temporal.weight)
        nn.init.zeros_(self.W_temporal.bias)

        self.W_condition = nn.Linear(768 * 3 + V_DIM, 768, bias=True)
        nn.init.xavier_uniform_(self.W_condition.weight)
        nn.init.zeros_(self.W_condition.bias)

        self.W_causal = nn.Linear(768 * 3 + V_DIM, 768, bias=True)
        nn.init.xavier_uniform_(self.W_causal.weight)
        nn.init.zeros_(self.W_causal.bias)

        # [NEW] emb(verdict) inside the verification vector v
        self.verdict_embedding = nn.Embedding(len(VERDICTS), VERDICT_EMB_DIM)

        # [NEW] gated variant: g = sigmoid(W_g v), e = g * W^r_sup(.) + (1 - g) * W^r_unsup(.)
        if USE_GATED_ENCODER:
            self.W_g = nn.Linear(V_DIM, 768, bias=True)
            nn.init.xavier_uniform_(self.W_g.weight)
            nn.init.zeros_(self.W_g.bias)

            self.W_unsup = nn.ModuleDict()
            for relation_name in RELATION_NAMES[1:]:
                self.W_unsup[relation_name] = nn.Linear(768 * 3 + V_DIM, 768, bias=True)
                nn.init.xavier_uniform_(self.W_unsup[relation_name].weight)
                nn.init.zeros_(self.W_unsup[relation_name].bias)

        self.sigmoid = nn.Sigmoid()

    def token_embedding(self, roberta_input_ids, roberta_attention_mask): # input size: batch size * number of tokens

        outputs = self.roberta(input_ids=roberta_input_ids, attention_mask=roberta_attention_mask)
        hidden_states = outputs[2]
        token_embeddings_layers = torch.stack(hidden_states, dim=0)  # 13 layer * batch_size * number of tokens * 768
        token_embeddings = torch.sum(token_embeddings_layers[-4:, :, :, :], dim=0) # sum up the last four layers, batch_size * number of tokens * 768

        return token_embeddings

    def sequence_embedding(self, roberta_input_ids, roberta_attention_mask):

        token_embeddings_in_sequence = self.token_embedding(roberta_input_ids, roberta_attention_mask) # batch size * number of tokens * 768
        sum_token_embeddings = torch.sum(torch.mul(token_embeddings_in_sequence, roberta_attention_mask.view(roberta_attention_mask.shape[0], roberta_attention_mask.shape[1], 1).repeat(1, 1, 768)), dim=1)
        num_none_padding_tokens = torch.sum(roberta_attention_mask, dim=1).view(roberta_attention_mask.shape[0], 1).repeat(1, 768)
        meal_pooling_embedding = torch.div(sum_token_embeddings, num_none_padding_tokens) # batch size * 768

        return meal_pooling_embedding

    def verification_vector(self, nli_scores, verdict_ids, verdict_conf, flags):
        ''' [NEW] v = [P_NLI(a->b) ; P_NLI(b->a) ; emb(verdict) ; conf] from Stage 2, plus the Stage 3 flag '''

        if not USE_VERDICT_VECTOR: # [FIX] ablation / control: v carries no verification information
            return torch.zeros((nli_scores.shape[0], V_DIM), device=nli_scores.device)

        return torch.cat((nli_scores, self.verdict_embedding(verdict_ids), verdict_conf.view(-1, 1), flags.view(-1, 1)), dim=1) # number of relations * V_DIM

    def subtree_embedding(self, left_argument, logical_keyword, right_argument, verification, logical_relation):
        ''' [CHANGED] e = W^r(e_l + e_c + e_r + v) + b^r; e_l / e_r are already replaced by subtree embeddings
        by the caller (bottom-up composition), so the original lower_subtree fusion is no longer done here '''

        concat_embedding = torch.cat((left_argument, logical_keyword, right_argument, verification), dim=1)

        if logical_relation == 1:
            encoded = self.W_conjuction(concat_embedding)
        if logical_relation == 2:
            encoded = self.W_alternative(concat_embedding)
        if logical_relation == 3:
            encoded = self.W_restatement(concat_embedding)
        if logical_relation == 4:
            encoded = self.W_instantiation(concat_embedding)
        if logical_relation == 5:
            encoded = self.W_contrast(concat_embedding)
        if logical_relation == 6:
            encoded = self.W_concession(concat_embedding)
        if logical_relation == 7:
            encoded = self.W_analogy(concat_embedding)
        if logical_relation == 8:
            encoded = self.W_temporal(concat_embedding)
        if logical_relation == 9:
            encoded = self.W_condition(concat_embedding)
        if logical_relation == 10:
            encoded = self.W_causal(concat_embedding)

        # [NEW] gated variant
        if USE_GATED_ENCODER:
            gate = self.sigmoid(self.W_g(verification))
            encoded_unsup = self.W_unsup[RELATION_NAMES[logical_relation]](concat_embedding)
            encoded = gate * encoded + (1 - gate) * encoded_unsup

        return encoded

    def compose_sentence_tree(self, left_argument_embedding, logical_keyword_embedding, right_argument_embedding, verification, index_logical_relation, parent_index, parent_side):
        ''' [NEW] bottom-up composition over the real tree structure: the embeddings of child subtrees replace
        e_l (children inside alpha) or e_r (children inside beta) of their parent; several children on one side are averaged.
        Returns the root embedding (averaged if a sentence has several root relations). '''

        num_relations = left_argument_embedding.shape[0]
        parents = parent_index.tolist()
        sides = parent_side.tolist()
        relations = index_logical_relation.tolist()
        depth = node_depths(parents)

        subtree = [None] * num_relations
        for i in sorted(range(num_relations), key=lambda x: -depth[x]): # deepest first
            alpha_children = [subtree[j] for j in range(num_relations) if parents[j] == i and sides[j] == 0]
            beta_children = [subtree[j] for j in range(num_relations) if parents[j] == i and sides[j] == 1]

            e_l = torch.stack(alpha_children, dim=0).mean(dim=0) if len(alpha_children) != 0 else left_argument_embedding[i, :].view(1, 768)
            e_r = torch.stack(beta_children, dim=0).mean(dim=0) if len(beta_children) != 0 else right_argument_embedding[i, :].view(1, 768)

            subtree[i] = self.subtree_embedding(e_l, logical_keyword_embedding[i, :].view(1, 768), e_r, verification[i, :].view(1, V_DIM), relations[i])

        roots = [subtree[i] for i in range(num_relations) if parents[i] == -1]
        return torch.stack(roots, dim=0).mean(dim=0) # 1 * 768

    def forward(self, logical_keyword_input_ids, logical_keyword_attention_mask, left_argument_input_ids, left_argument_attention_mask, right_argument_input_ids, right_argument_attention_mask, sentence_text_input_ids, sentence_text_attention_mask, num_logical_relations_each_sent, index_logical_relation,
                parent_index, parent_side, nli_scores, verdict_ids, verdict_conf, flags): # [CHANGED] new inputs

        tree_embedding = torch.zeros((1, 768)).to(device)

        if sentence_text_input_ids.shape[0] != 0:
            sentence_text_embedding = self.sequence_embedding(sentence_text_input_ids, sentence_text_attention_mask)  # number of sentences * 768
            num_sents = sentence_text_embedding.shape[0]

            if logical_keyword_input_ids.shape[0] != 0:
                logical_keyword_embedding = self.sequence_embedding(logical_keyword_input_ids, logical_keyword_attention_mask)  # number of logical relations * 768
                left_argument_embedding = self.sequence_embedding(left_argument_input_ids, left_argument_attention_mask)
                right_argument_embedding = self.sequence_embedding(right_argument_input_ids, right_argument_attention_mask)
                verification = self.verification_vector(nli_scores, verdict_ids, verdict_conf, flags) # [NEW] number of logical relations * V_DIM

                for sent_i in range(num_sents):
                    num_logical_relations_this_sent = num_logical_relations_each_sent[sent_i].item()

                    start_index_of_logical_relation = 0 if sent_i == 0 else torch.sum(num_logical_relations_each_sent[:sent_i]).item()
                    end_index_of_logical_relation = torch.sum(num_logical_relations_each_sent[:(sent_i + 1)]).item()

                    logical_keyword_embedding_this_sent = logical_keyword_embedding[start_index_of_logical_relation:end_index_of_logical_relation, :]
                    left_argument_embedding_this_sent = left_argument_embedding[start_index_of_logical_relation:end_index_of_logical_relation, :]
                    right_argument_embedding_this_sent = right_argument_embedding[start_index_of_logical_relation:end_index_of_logical_relation, :]
                    index_logical_relation_this_sent = index_logical_relation[start_index_of_logical_relation:end_index_of_logical_relation]

                    if num_logical_relations_this_sent != 0:
                        # [CHANGED] bottom-up composition over the tree structure (replaces the original right-argument chain)
                        sentence_tree_embedding = self.compose_sentence_tree(left_argument_embedding_this_sent, logical_keyword_embedding_this_sent, right_argument_embedding_this_sent,
                                                                             verification[start_index_of_logical_relation:end_index_of_logical_relation, :],
                                                                             index_logical_relation_this_sent,
                                                                             parent_index[start_index_of_logical_relation:end_index_of_logical_relation],
                                                                             parent_side[start_index_of_logical_relation:end_index_of_logical_relation])

                        tree_embedding += self.W_text_tree(torch.cat((sentence_text_embedding[sent_i, :].view(1, 768), sentence_tree_embedding), dim=1))

                    else:
                        tree_embedding += sentence_text_embedding[sent_i, :].view(1, 768)

            else:
                for sent_i in range(num_sents):
                    tree_embedding += sentence_text_embedding[sent_i, :].view(1, 768)

            tree_embedding = tree_embedding / num_sents

        # two-layer projection into the LLM embedding space (unchanged)
        tree_embedding = self.projection_layer_2(self.sigmoid(self.projection_layer_1(tree_embedding)))

        return tree_embedding



class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        # Stage 5: Llama-2-7B + LoRA, base frozen (unchanged)
        self.llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16)
        lora_config = LoraConfig(r=lora_rank, target_modules=['q_proj', 'v_proj'], lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
        self.llama = get_peft_model(self.llama, lora_config)

        self.tree = Tree_Embedding()

    def forward(self, source_input_ids, target_input_ids, llama_soft_token, logical_keyword_input_ids, logical_keyword_attention_mask, left_argument_input_ids, left_argument_attention_mask, right_argument_input_ids, right_argument_attention_mask, sentence_text_input_ids, sentence_text_attention_mask, num_logical_relations_each_sent, index_logical_relation,
                parent_index, parent_side, nli_scores, verdict_ids, verdict_conf, flags, inference_mode): # [CHANGED] new inputs

        tree_embedding = self.tree(logical_keyword_input_ids, logical_keyword_attention_mask, left_argument_input_ids, left_argument_attention_mask, right_argument_input_ids, right_argument_attention_mask, sentence_text_input_ids, sentence_text_attention_mask, num_logical_relations_each_sent, index_logical_relation,
                                   parent_index, parent_side, nli_scores, verdict_ids, verdict_conf, flags) # [CHANGED]
        tree_embedding = tree_embedding.view(1, tree_embedding.shape[0], tree_embedding.shape[1])

        source_input_embeds = self.llama.model.model.embed_tokens(source_input_ids)
        target_input_embeds = self.llama.model.model.embed_tokens(target_input_ids)

        tree_embedding = tree_embedding.to(source_input_embeds.dtype)

        ''' add tree embedding as soft token '''

        new_source_input_embeds = source_input_embeds[:, :llama_soft_token.item(), :]
        new_source_input_embeds = torch.cat((new_source_input_embeds, tree_embedding), dim=1)
        new_source_input_embeds = torch.cat((new_source_input_embeds, source_input_embeds[:, llama_soft_token.item():, :]), dim=1)

        ''' form sequence and label '''
        sequence_input_embeds = torch.cat((new_source_input_embeds, target_input_embeds), dim=1)
        label_input_ids = torch.cat((torch.full((new_source_input_embeds.shape[0], new_source_input_embeds.shape[1]), -100, dtype=torch.int64).to(device), target_input_ids), dim=1)

        if inference_mode == 1:
            outputs = self.llama.generate(inputs_embeds=new_source_input_embeds, max_new_tokens=max_target_length)
            return outputs
        else:
            loss = self.llama(inputs_embeds=sequence_input_embeds, labels=label_input_ids).loss
            return loss


    def print_trainable_params(self):

        trainable_params = 0
        all_params = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_params






''' evaluate '''

def convert_label(text):

    for label_i in range(len(fallacy_list)):
        if fallacy_list[label_i] in text:
            return label_i

    return -1


def evaluate(model, eval_dataloader, verbose, output_path=None): # [CHANGED] optional output file for labels + explanations

    model.eval()

    true_label = []
    prediction = []

    step_error = []

    strict_true = []   # [FIX] strict protocol: every example is scored; invalid outputs / errors count as wrong
    strict_pred = []

    output_records = [] # [NEW] Stage 5 outputs: label from the LLM, explanation assembled from the verified tree

    for step, batch in enumerate(eval_dataloader):

        source_input_ids = batch["source_input_ids"][0]
        target_input_ids = batch["target_input_ids"][0]
        llama_soft_token = batch["llama_soft_token"][0]
        logical_keyword_input_ids = batch["logical_keyword_input_ids"][0]
        logical_keyword_attention_mask = batch["logical_keyword_attention_mask"][0]
        left_argument_input_ids = batch["left_argument_input_ids"][0]
        left_argument_attention_mask = batch["left_argument_attention_mask"][0]
        right_argument_input_ids = batch["right_argument_input_ids"][0]
        right_argument_attention_mask = batch["right_argument_attention_mask"][0]
        sentence_text_input_ids = batch["sentence_text_input_ids"][0]
        sentence_text_attention_mask = batch["sentence_text_attention_mask"][0]
        num_logical_relations_each_sent = batch["num_logical_relations_each_sent"][0]
        index_logical_relation = batch["index_logical_relation"][0]
        # [NEW]
        parent_index = batch["parent_index"][0].to(device)
        parent_side = batch["parent_side"][0].to(device)
        nli_scores = batch["nli_scores"][0].to(device)
        verdict_ids = batch["verdict_ids"][0].to(device)
        verdict_conf = batch["verdict_conf"][0].to(device)
        flags = batch["flags"][0].to(device)

        source_input_ids, target_input_ids, llama_soft_token, logical_keyword_input_ids, logical_keyword_attention_mask, left_argument_input_ids, left_argument_attention_mask, right_argument_input_ids, right_argument_attention_mask, sentence_text_input_ids, sentence_text_attention_mask, num_logical_relations_each_sent, index_logical_relation = \
            source_input_ids.to(device), target_input_ids.to(device), llama_soft_token.to(device), logical_keyword_input_ids.to(device), logical_keyword_attention_mask.to(device), left_argument_input_ids.to(device), left_argument_attention_mask.to(device), right_argument_input_ids.to(device), right_argument_attention_mask.to(device), sentence_text_input_ids.to(device), sentence_text_attention_mask.to(device), num_logical_relations_each_sent.to(device), index_logical_relation.to(device)

        label_text = llama_tokenizer.decode(target_input_ids[0])
        gold = convert_label(label_text)       # [FIX]
        predicted_label = -1                   # [FIX] -1 = invalid output or error
        generated_text = ""                    # [FIX]

        try:
            # inference
            with torch.no_grad():
                outputs = model(source_input_ids, target_input_ids, llama_soft_token, logical_keyword_input_ids, logical_keyword_attention_mask,
                                left_argument_input_ids, left_argument_attention_mask, right_argument_input_ids, right_argument_attention_mask,
                                sentence_text_input_ids, sentence_text_attention_mask, num_logical_relations_each_sent, index_logical_relation,
                                parent_index, parent_side, nli_scores, verdict_ids, verdict_conf, flags, inference_mode=1) # [CHANGED]

            generated_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_label = convert_label(generated_text)
            if predicted_label == -1:
                step_error.append(step)
            else:
                prediction.append(predicted_label)
                true_label.append(gold)

        except Exception as e: # [FIX] never hide a real bug: print the first few errors
            step_error.append(step)
            if len(step_error) <= 3:
                print("evaluation error at step", step, ":", repr(e))

        strict_true.append(gold)                # [FIX]
        strict_pred.append(predicted_label)     # [FIX]

        # [NEW] [FIX] Stage 5 record for EVERY example (aligned with the test set, needed for significance tests)
        if output_path is not None:
            output_records.append({"index": step,
                                   "text": batch["text"][0],
                                   "gold_label": label_text.replace("</s>", "").strip(),
                                   "predicted_label": fallacy_list[predicted_label] if predicted_label != -1 else "INVALID",
                                   "raw_output": generated_text.strip(),
                                   "tree_level_summary": json.loads(batch["tree_level_json"][0]),
                                   "tree_summary": batch["tree_summary"][0],
                                   "explanation": batch["explanation"][0]})


    if len(step_error) != 0:
        print("step error is ", len(step_error))

    # [NEW] write labels + explanations
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            for record in output_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if len(prediction) != 0:

        macro_precision = precision_recall_fscore_support(true_label, prediction, average='macro')[0]
        macro_recall = precision_recall_fscore_support(true_label, prediction, average='macro')[1]
        macro_F = precision_recall_fscore_support(true_label, prediction, average='macro')[2]
        micro_F = precision_recall_fscore_support(true_label, prediction, average='micro')[2]

        if verbose:
            print("Macro: ", precision_recall_fscore_support(true_label, prediction, average='macro'))
            print("Micro: ", precision_recall_fscore_support(true_label, prediction, average='micro'))
            # print("Classification Report: \n", classification_report(true_label, prediction, digits=4))

    else:
        macro_precision = 0
        macro_recall = 0
        macro_F = 0
        micro_F = 0

    # [FIX] strict protocol (errors counted as wrong), reported next to the original protocol
    all_labels = list(range(len(fallacy_list)))
    strict = precision_recall_fscore_support(strict_true, strict_pred, labels=all_labels, average='macro', zero_division=0)
    strict_accuracy = accuracy_score(strict_true, strict_pred)
    if verbose:
        print("Strict macro (errors counted as wrong): ", strict)
        print("Strict accuracy: ", strict_accuracy)
        print("Per-class F1 (strict):")
        per_class = precision_recall_fscore_support(strict_true, strict_pred, labels=all_labels, average=None, zero_division=0)[2]
        for label_i in all_labels:
            print("   {:<26s} {:.4f}".format(fallacy_list[label_i], per_class[label_i]))

    evaluate.last_strict = {"strict_macro_precision": strict[0], "strict_macro_recall": strict[1], "strict_macro_F": strict[2],
                            "strict_accuracy": strict_accuracy, "num_errors": len(step_error), "num_examples": len(strict_true)} # [FIX]

    return macro_precision, macro_recall, macro_F, micro_F








''' train '''

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()



random.seed(seed_val) # [CHANGED] seed_val is now set from the command line (see the top of the file)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.use_deterministic_algorithms(True, warn_only=True)


model = Model()
model.cuda()
trainable_params, all_params = model.print_trainable_params()
print("all_params is {:}, trainable_params is {:}, ratio of trainable_params is {:}".format(all_params, trainable_params, 100 * trainable_params / all_params))


param_all = list(model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('roberta' in n))], 'lr': roberta_lr, 'weight_decay': weight_decay},
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('tree' in n) and (not 'roberta' in n))], 'lr': tree_lr, 'weight_decay': weight_decay},
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('llama' in n))], 'lr': llama_lr, 'weight_decay': weight_decay},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('roberta' in n))], 'lr': roberta_lr, 'weight_decay': 0.0},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('tree' in n) and (not 'roberta' in n))], 'lr': tree_lr, 'weight_decay': 0.0},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('llama' in n))], 'lr': llama_lr, 'weight_decay': 0.0}]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8)


train_dataset = custom_dataset(train, verified_trees["train"]) # [CHANGED]
dev_dataset = custom_dataset(dev, verified_trees["dev"])       # [CHANGED]
test_dataset = custom_dataset(test, verified_trees["test"])    # [CHANGED]

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


num_train_steps = num_epochs * len(train_dataloader) // gradient_accumulation_steps # scheduler.step_with_optimizer = True by default
warmup_steps = int(warmup_proportion * num_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)


accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)


best_macro_F_dev = 0

for epoch_i in range(num_epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i, num_epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0
    num_batch = 0  # number of batch to calculate average loss
    total_num_batch = 0  # number of batch in this epoch

    for batch in train_dataloader:

        if total_num_batch % valid_steps == 0 and total_num_batch != 0:

            # valid every valid_steps, actual update steps = valid_steps / gradient_accumulation_steps

            elapsed = format_time(time.time() - t0)
            avg_loss = total_loss / num_batch if num_batch != 0 else 0
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    loss average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_loss))

            total_loss = 0
            num_batch = 0

            macro_precision, macro_recall, macro_F, micro_F = evaluate(model, dev_dataloader, verbose=0)

            if macro_F > best_macro_F_dev:
                torch.save(model.state_dict(), checkpoint_path) # [CHANGED] per-experiment checkpoint
                best_macro_F_dev = macro_F


        model.train()

        source_input_ids = batch["source_input_ids"][0]
        target_input_ids = batch["target_input_ids"][0]
        llama_soft_token = batch["llama_soft_token"][0]
        logical_keyword_input_ids = batch["logical_keyword_input_ids"][0]
        logical_keyword_attention_mask = batch["logical_keyword_attention_mask"][0]
        left_argument_input_ids = batch["left_argument_input_ids"][0]
        left_argument_attention_mask = batch["left_argument_attention_mask"][0]
        right_argument_input_ids = batch["right_argument_input_ids"][0]
        right_argument_attention_mask = batch["right_argument_attention_mask"][0]
        sentence_text_input_ids = batch["sentence_text_input_ids"][0]
        sentence_text_attention_mask = batch["sentence_text_attention_mask"][0]
        num_logical_relations_each_sent = batch["num_logical_relations_each_sent"][0]
        index_logical_relation = batch["index_logical_relation"][0]
        # [NEW]
        parent_index = batch["parent_index"][0].to(device)
        parent_side = batch["parent_side"][0].to(device)
        nli_scores = batch["nli_scores"][0].to(device)
        verdict_ids = batch["verdict_ids"][0].to(device)
        verdict_conf = batch["verdict_conf"][0].to(device)
        flags = batch["flags"][0].to(device)

        with accelerator.accumulate(model):

            loss = model(source_input_ids, target_input_ids, llama_soft_token, logical_keyword_input_ids, logical_keyword_attention_mask,
                         left_argument_input_ids, left_argument_attention_mask, right_argument_input_ids, right_argument_attention_mask,
                         sentence_text_input_ids, sentence_text_attention_mask, num_logical_relations_each_sent, index_logical_relation,
                         parent_index, parent_side, nli_scores, verdict_ids, verdict_conf, flags, inference_mode=0) # [CHANGED]

            total_loss += loss.item()
            num_batch += 1
            total_num_batch += 1

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()



    # valid at the end of each epoch

    elapsed = format_time(time.time() - t0)
    avg_loss = total_loss / num_batch if num_batch != 0 else 0
    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    loss average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_loss))

    total_loss = 0
    num_batch = 0

    macro_precision, macro_recall, macro_F, micro_F = evaluate(model, dev_dataloader, verbose=0)

    if macro_F > best_macro_F_dev:
        torch.save(model.state_dict(), checkpoint_path) # [CHANGED]
        best_macro_F_dev = macro_F

if not os.path.exists(checkpoint_path): # [FIX] e.g. a smoke run where the dev score stayed 0: test the final model
    torch.save(model.state_dict(), checkpoint_path)



# test

model.load_state_dict(torch.load(checkpoint_path, map_location=device)) # [CHANGED]
test_output_path = os.path.join(OUTPUT_DIR, dataset_name + "_test_predictions_" + RUN_NAME + ".jsonl") # [FIX] one file per experiment
macro_precision, macro_recall, macro_F, micro_F = evaluate(model, test_dataloader, verbose=1, output_path=test_output_path) # [CHANGED] writes labels + explanations


# [FIX] append this run's results to one log, so all experiments can be compared in a table later
results = {"run": RUN_NAME, "dataset": dataset_name, "mode": VERIFIER_MODE, "gated": USE_GATED_ENCODER,
           "verdict_text": USE_VERDICT_TEXT, "verdict_vector": USE_VERDICT_VECTOR, "shuffled": SHUFFLE_VERDICTS, "seed": seed_val,
           "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_F": macro_F, "accuracy": micro_F,
           "best_dev_macro_F": best_macro_F_dev, **evaluate.last_strict,
           "nli_model": NLI_MODEL_NAME, "judge_model": JUDGE_MODEL_NAME,
           "nli_threshold": NLI_THRESHOLD, "nli_circular_threshold": NLI_CIRCULAR_THRESHOLD,
           "finished": datetime.datetime.now().isoformat(timespec="seconds")}
with open(os.path.join(OUTPUT_DIR, "results_" + dataset_name + ".jsonl"), "a", encoding="utf-8") as f:
    f.write(json.dumps(results) + "\n")
print("Results appended to", os.path.join(OUTPUT_DIR, "results_" + dataset_name + ".jsonl"))


# [FIX] each checkpoint is ~14GB; the predictions and results are saved, so delete it unless asked to keep it
if not args.keep_checkpoint:
    os.remove(checkpoint_path)
    print("Checkpoint deleted (use --keep_checkpoint to keep it).")













# stop here
