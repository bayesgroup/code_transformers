# This script selects all top-level functions from code files,
# filters functions
# and saves asts and preprocessed data

import re
import json
import os
import random
import numpy as np
import argparse
import string
from collections import Counter

letters = set(string.ascii_letters)

EMPTY_VAL = "<emptyvalue>"
FUN_VAL = "<fun_name>"
FUNCTION_TYPES_JS = {"FunctionDeclaration", "FunctionExpression"}
FUNCTION_TYPES_PY = {"FunctionDef"}
NAMED_FUNCTIONEXPR_TYPES_JS = {"AssignmentExpression", "Property",
                                "MemberExpression", "VariableDeclarator"}
VAR_TYPES_PY = {"NameParam", "NameStore", "NameLoad"}
VAR_TYPES_JS = {"Identifier", "Property", "VariableDeclarator"}
ANO_PATTERN = "<var%d>"

AST_FILE_FORMAT = "asts_%s.json"
FN_FILE_FORMAT = "filenames_%s.txt"
TYPE_FILE_FORMAT = "traverse_types_%s.txt"
VALUE_FILE_FORMAT = "traverse_values_%s.txt"
ANOVALUE_FILE_FORMAT = "traverse_anovalues_%s.txt"
TARGET_FILE_FORMAT = "targets_%s.txt"
ONLYVALUES_FILE_FORMAT = "traverse_onlyvalues_%s.txt"
ONLYVALUES_TARGET_FILE_FORMAT = "targets_onlyvalues_%s.txt"

def unflatten_tree(dfs):
    new_dfs = []
    for node in dfs:
        new_dfs.append(node.copy())
    for i, node in enumerate(new_dfs):
        if "children" in node:
            for ch in node["children"]:
                new_dfs[ch]["parent"] = i
    result = []
    for i, node in enumerate(new_dfs):
        tree_node = {}
        tree_node["type"] = node["type"]
        if "value" in node:
            tree_node["value"] = node["value"]
        tree_node["parent_type"] = new_dfs[node["parent"]]["type"] \
                           if i > 0 else None # needed for JS FN
        # prev_value needed for JS FN
        if i > 0 and "value" in new_dfs[i-1]:
            tree_node["prev_value"] = new_dfs[i-1]["value"]
        else:
            tree_node["prev_value"] = None
        tree_node["children"] = []
        result.append(tree_node)
        if "parent" in node:
            result[node["parent"]]["children"].append(tree_node)
    return result[0]

def flatten_tree(tree):
    dfs = []
    def get_node(tree_node):
        node = {}
        node["type"] = tree_node["type"]
        if "value" in tree_node:
            node["value"] = tree_node["value"]
        return node
            
    def process_node(tree_node):
        node = get_node(tree_node)
        idx = len(dfs)
        dfs.append(node)
        if len(tree_node["children"]):
            node["children"] = []
            for child in tree_node["children"]:
                node["children"].append(process_node(child))
        return idx
    
    process_node(tree)
    return dfs

def get_function_name(node, lang, task):
    """
    Extracts function names: PY: FN is always stored in the first node.
    JS: for FunctionDeclaration, FN is always stored in the first child of the root node.
    FunctionExpressions are anonymous, thus we selected the types of parent nodes in which the function is assigned to some name, e. g. var = <function> or key:<function>, and use these names.
    Other FunctionExpressions are filtered out.
    """
    def replace_and_return_name(node):
        function_name = node["value"]
        if task == "fn":
            node["value"] = FUN_VAL
        return function_name
    
    if lang == "py":
        return replace_and_return_name(node)
    else: # "js"
        if node["type"] == "FunctionDeclaration":
            return replace_and_return_name(node["children"][0])
        else: # FunctionExpression
            # anonymous --- function name is not inside function
            if node["parent_type"] in NAMED_FUNCTIONEXPR_TYPES_JS:
                if type(node["prev_value"]) == str:
                    return node["prev_value"]
                else:
                    return None
            else:
                return None

def select_functions(tree, lang, task):
    """
    tree: unflattened tree
    lang: "py" or "js"
    function selects all top-level functions from the file
    (functions inside classes are also included)
    """
    function_types = FUNCTION_TYPES_PY if lang == "py" else FUNCTION_TYPES_JS
    functions = []
    stack = [tree]
    while len(stack):
        node = stack.pop()
        if node["type"] in function_types:
            function_name = get_function_name(node, lang, task)
            functions.append((node, function_name))
        else:
            for child in node["children"][::-1]:
                stack.append(child)
    return functions

def js_property_filter(value):
    ### returns True if value is a valid JS identifier
    ### and False otherwise (value is a string/integer)
    return type(value) == str and value.replace("$", "_").isidentifier()

def get_positions(fun, lang):
    """
    function selects all variable positions in fun (for VM task)
    """
    var_positions = []
    vars_ = {}
    var_types = VAR_TYPES_PY if lang=="py" else VAR_TYPES_JS
    if lang == "py":
        node_filter = lambda node: node["type"] in VAR_TYPES_PY
    else:
        node_filter = lambda node: node["type"] in VAR_TYPES_JS and \
                                   "value" in node and \
                                   (node["type"] != "Property" or \
                                    js_property_filter(node["value"]))
    # last condition ensures that JS properties that are not valid identifiers
    #    are not included in bug generation or fixing and are treated as strings
    # for some reason, there are rare nodes in JS that are in VAR_TYPES_JS
    #    and do not contain value
        
    for i, node in enumerate(fun):
        if node_filter(node): 
            var_positions.append(i)
            if not node["value"] in vars_:
                vars_[node["value"]] = 0
            vars_[node["value"]] += 1
    allowed_var_positions = [pos for pos in var_positions \
                             if vars_[fun[pos]["value"]]>1]
    return vars_, var_positions, allowed_var_positions

def filter_out_varmisuse(fun, lang, max_len):
    """
    filter out functions for which any condition is not true:
    * len < max_len
    * number of different vars >= 3 (otherwise fix is trivial)
    * number of different var positions >= 3 (otherwise fix is trivial)
    * there is at least one repeating variable (not repeating vars couldn't be fixed with pointer)
    
    fun: dfs (same format as in json data file)
    lang: "py"|"js"
    max_len: int
    
    """
    if len(fun) > max_len:
        return True
    vars_, var_positions, allowed_var_positions = get_positions(fun, lang)
    return len(vars_) < 3 or len(var_positions) < 3 or \
           len(allowed_var_positions) < 1

def tokenize_with_camel_case(token):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token)
    return [m.group(0) for m in matches]

def process_token(token):
    """
    OneTwo -> one_two
    x.y.x -> x_y_z
    x__z -> x_z
    """
    token = "_".join(tokenize_with_camel_case(token))
    token = "_".join(token.split("."))
    token = "_".join(token.strip("_").split("_"))
    if token == "":
        return "_"
    else:
        return token.lower()

def get_filterfun_funnaming(funs, tgt_vocab_size, max_len):
    letters = set(string.ascii_letters)
    tgt_counts = Counter()
    for fun, filename, function_name in funs:
        if len(fun) <= max_len and function_name is not None and \
           any([let in letters for let in function_name]):
            name = process_token(function_name)
            tgt_counts.update(name.split("_"))
    tgt_vocab = set([elem[0] for elem in tgt_counts.most_common(tgt_vocab_size)])
    
    def filter_out_funnaming(fun, function_name):
        """
        filter out functions for which any condition is not true:
        * len < max_len
        * name is not None (never true for PY)
        * all tokens in the function name are in tgt_vocab (otherwise it could be impossible to predict the full name)
        * the function name includes at least one letter (to avoid names like __)
        
        fun: dfs (same format as in json data file)
        lang: "py"|"js"
        max_len: int
        min_tgt_token_freq: int

        """
        if len(fun) > max_len or function_name is None:
            # last conditions filters out JS functions with no name
            return True
        name = process_token(function_name)
        return any([w not in tgt_vocab for w in name.split("_")]) \
               or all([let not in letters for let in name])
    return filter_out_funnaming

CHARS = string.ascii_letters + string.digits + string.punctuation + " "
def isnormal(s):
    return all(c in CHARS for c in s) and not s==""

def convert_value_js(elem, is_fn_task):
    """
    preprocessing values before they are written into file
    elem in string (value itself)
    is_fn_task: True for task=="fn" else False
    """
    if not "value" in elem:
        return EMPTY_VAL
    elif elem["type"] == "LiteralRegExp":
        return ("<regexp>" if (not is_fn_task or not isnormal(elem["value"])) \
                          else elem["value"].replace(" ", "_"))
    elif elem["type"] == "LiteralString":
        return ("<string>" if (not is_fn_task or not isnormal(elem["value"])) \
                          else elem["value"].replace(" ", "_"))
    elif elem["type"] == "Property" and not js_property_filter(elem["value"]):
        # last condition means that is property is a valid identifier
        # it is processed below with identifiers
        return ("<property>" if (not is_fn_task or \
                                 not isnormal(str(elem["value"]))) \
                          else str(elem["value"]).replace(" ", "_"))
    else:
        return elem["value"] if not is_fn_task else process_token(elem["value"])
        
def convert_value_py(elem, is_fn_task):
    """
    preprocessing values before they are written into file
    elem is a string (value itself)
    i is position in dfs (needed only for function naming)
    is_fn_task: True for task=="fn" else False
    """
    if not "value" in elem:
        return EMPTY_VAL
    elif elem["type"] == "Str":
        return ("<string>" if (not is_fn_task or not isnormal(elem["value"])) \
                          else elem["value"].replace(" ", "_"))
    else:
        return elem["value"] if not is_fn_task else process_token(elem["value"])
        
def gen_examples(fun, lang, max_bugs_per_fun):
    """
    VM task: for each function, generates synthetic bugs
    each bug is a tuple of 3 elements:
    - position of the bug, 
    - position of some other variable to be inserted at the bug position, 
    - positions to fix bug, i. e. all other positions with the same variable as in the bug position
    
    fun: dfs (same format as in json data file)
    lang: "py"|"js"
    max_bugs_per_fun: int
    """
    vars_, var_positions, allowed_var_positions = get_positions(fun, lang)
    target = []
    used_positions = set()
    for _ in range(min(len(allowed_var_positions), max_bugs_per_fun)):
        bug_pos = random.choice([pos for pos in allowed_var_positions \
                                    if not pos in used_positions])
        used_positions.add(bug_pos)
        true_var = fun[bug_pos]["value"]
        true_other_pos = []
        false_other_pos = []
        for pos in var_positions:
            if fun[pos]["value"] != true_var:
                false_other_pos.append(pos)
            elif pos != bug_pos:
                true_other_pos.append(pos)
        replace_pos = random.choice(false_other_pos)
        target.append([bug_pos, replace_pos, true_other_pos])
    
    return var_positions, target

def convert_target(var_positions, target):
    """
    function for saving targets into a file
    """
    def convert_poss(poss):
        pos, bug, fixes = poss
        return str(pos)+"_"+str(bug)+"_"+\
               "|".join([str(fix) for fix in fixes])
    return "_".join([str(pos) for pos in var_positions])+" "+\
           " ".join([convert_poss(targ_elem) \
                     for targ_elem in target])

def get_onlyvalues_target(ast, var_positions, targets):
    """
    Renumbers target positions when deleting empty values
    """
    mapping = {}
    k = 0
    for i, node in enumerate(ast):
        if "value" in node:
            mapping[i] = k
            k += 1
    new_var_positions = [mapping[pos] for pos in var_positions]
    new_targets = [[mapping[pos1], mapping[pos2], \
                   [mapping[pos] for pos in poss3]] for pos1, pos2, poss3\
                                                    in targets]
    return new_var_positions, new_targets
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for Variable Misuse or Function naming tasks")
    parser.add_argument("--base_dir", type=str, default="../data/")
    parser.add_argument("--output_dir", type=str, default="../preprocessed_data/")
    parser.add_argument("--lang", type=str, default="py", help="py|js")
    parser.add_argument("--task", type=str, default="vm", help="vm|fn, vm = variable misuse, fn = function naming")
    parser.add_argument("--max_len", type=int, default=250, help="max len of function, used in finction filtering")
    parser.add_argument("--tgt_vocab_size", type=int, default=50000, help="functions with names with any word being not in tgt_vocab would be filtered out")
    parser.add_argument("--generate_target", type=bool, default=False, help="default False because downloading authors' target is suggested")
    parser.add_argument("--max_bugs_per_fun", type=int, default=3, help="needed for target generation")
    args = parser.parse_args()
    assert args.task in {"vm", "fn"}, "Task should be either vm or fn"
    assert args.lang in {"py", "js"}, "Lang should be either py or js"

    args.train = "train_%s.dedup" % args.lang
    args.val = "val_%s.dedup" % args.lang
    args.test = "test_%s.dedup" % args.lang
    json_train = os.path.join(args.base_dir, args.train+".json")
    json_val = os.path.join(args.base_dir, args.val+".json")
    json_test = os.path.join(args.base_dir, args.test+".json")
    txt_train = os.path.join(args.base_dir, args.train+".txt")
    txt_val = os.path.join(args.base_dir, args.val+".txt")
    txt_test = os.path.join(args.base_dir, args.test+".txt")

    ### Step 1: gather functions from all files
    all_functions = {"train":[], "val":[], "test":[]}
    seen_funs = set()
    for json_file, txt_file, label in [[json_train, txt_train, "train"],\
                           [json_val, txt_val, "val"],\
                           [json_test, txt_test, "test"]]:
        with open(json_file) as lines, open(txt_file) as filenames:
            for line_index, (line, filename) in enumerate(zip(lines, filenames)):
                if line_index % 10000 == 0:
                    print ('Processing line: ', line_index)
                dfs = json.loads(line)
                dfs = dfs[:-1] if args.lang=="js" else dfs
                tree = unflatten_tree(dfs)
                functions = select_functions(tree, args.lang, args.task)
                for fun, function_name in functions:
                    fun_flatten = flatten_tree(fun)
                    fun_json = json.dumps(fun_flatten)
                    if not fun_json in seen_funs:
                        all_functions[label].append([fun_flatten, filename, \
                                                     function_name])
                    seen_funs.add(fun_json)
            
    output_dir = os.path.join(args.output_dir, "python" if args.lang=="py" else "js")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if args.task == "fn":
        filter_out_funnaming = get_filterfun_funnaming(all_functions["train"],\
                                                       args.tgt_vocab_size, \
                                                       args.max_len)

    for label in ["train", "val", "test"]:
        funs_partition = all_functions[label]
        ### Step 2: filter out functions that do not fulfill task-speciic conditions (see help in corresponding functions)
        print(label, "before filtering:", len(funs_partition))
        if args.task == "vm":
            data = [(fun, filename, function_name) for fun, filename, function_name in funs_partition if not filter_out_varmisuse(fun, args.lang, args.max_len)]
        else:
            data = [(fun, filename, function_name) for fun, filename, function_name in funs_partition if not filter_out_funnaming(fun, function_name)]
        print(label, "after filtering:", len(data))

        ### Step 3: save ASTs and filenames (for future usage) and preprocessed data (to use in Transformer)
        # save ASTs
        with open(os.path.join(output_dir, AST_FILE_FORMAT%label), "w") as fout:
            fout.write("\n".join([json.dumps(fun) for fun, _, _ in data]))
        # save filenames
        with open(os.path.join(output_dir, FN_FILE_FORMAT%label), "w") as fout:
            fout.write("".join([filename for _, filename, _ in data]))
        # save a file with types
        with open(os.path.join(output_dir, TYPE_FILE_FORMAT%label), "w") as fout:
            fout.write("\n".join([" ".join([elem["type"] for elem in fun])\
                                            for fun, _, _ in data]))
        
        # save a file with values
        convert_value = convert_value_py if args.lang=="py" else convert_value_js
        is_fn_task = (args.task=="fn")
        with open(os.path.join(output_dir, VALUE_FILE_FORMAT%label), "w") as fout:
            fout.write("\n".join([" ".join([convert_value(elem, is_fn_task)\
                                            for i, elem in enumerate(fun)])\
                                            for fun, _, _ in data]))
        # save a file with nonempty values - to be used without types
        with open(os.path.join(output_dir, ONLYVALUES_FILE_FORMAT%label), "w") \
            as fout:
            fout.write("\n".join([" ".join([convert_value(elem, is_fn_task)\
                                            for i, elem in enumerate(fun)\
                                            if "value" in elem])\
                                            for fun, _, _ in data]))
            
        # check values
        with open(os.path.join(output_dir, TYPE_FILE_FORMAT%label)) as fin_types, \
             open(os.path.join(output_dir, VALUE_FILE_FORMAT%label)) as fin_vals:
            for l_types, l_values in zip(fin_types, fin_vals):
                types = l_types.strip().split()
                values = l_values.strip().split()
                assert len(types) == len(values)
                
        # save a file with targets, if needed and FN task
        if args.generate_target and args.task == "fn":
            with open(os.path.join(output_dir, TARGET_FILE_FORMAT%label), "w") as fout:
                fout.write("\n".join([process_token(function_name).replace("_", " ")\
                                            for _, _, function_name in data]))
                
        
    # generate and save anonymized values
    random.seed(10)
    for label in ["train", "val", "test"]:
        # save a file with anonymized values  
        ano_vocab_size = args.max_len # number of different values couldn't be greater than max_len
        with open(os.path.join(output_dir, VALUE_FILE_FORMAT%label)) as fin,\
             open(os.path.join(output_dir, ANOVALUE_FILE_FORMAT%label), "w") as fout:
            first = True
            for line in fin:
                values = line.strip().split()
                toks = []
                for value in values:
                    if not value in toks and value != EMPTY_VAL and value!=FUN_VAL:
                        toks.append(value)
                ans = list(range(1, ano_vocab_size))
                random.shuffle(ans)
                ans = ans[:len(toks)]
                tok2id = {tok:i for tok, i in zip(toks, ans)}
                code = [ANO_PATTERN%tok2id[tok] if tok not in {EMPTY_VAL,FUN_VAL} \
                        else tok for tok in values]
                fout.write(("\n" if not first else "")+" ".join(code))
                assert len(code) == len(values)
                first = False
                
    # generate and save targets, if asked, for VM 
    # (code includes randomness so it is placed separately)
    if args.generate_target and args.task == "vm":
        random.seed(1)
        for label in ["train", "val", "test"]:
            with open(os.path.join(output_dir, AST_FILE_FORMAT%label)) as fin_ast,\
                 open(os.path.join(output_dir,\
                                   TARGET_FILE_FORMAT%label), "w") as fout,\
                 open(os.path.join(output_dir, \
                              ONLYVALUES_TARGET_FILE_FORMAT%label), "w") as fout2:
                first = True
                for line_ast in fin_ast:
                    ast = json.loads(line_ast)
                    var_positions, targets = gen_examples(ast, args.lang, \
                                             args.max_bugs_per_fun)
                    new_var_positions, new_targets = get_onlyvalues_target(ast,\
                                             var_positions, targets)
                    # when leaving only nonempty values, target positions shift to the left
                    if not first:
                        fout.write("\n")
                        fout2.write("\n")
                    fout.write(convert_target(var_positions, targets))
                    fout2.write(convert_target(new_var_positions, new_targets))
                    first = False
    elif args.task == "vm": # check targets if not generate target
        for label in ["train", "val", "test"]:
            with open(os.path.join(output_dir, TARGET_FILE_FORMAT%label)) as fin_tgt,\
                 open(os.path.join(output_dir, VALUE_FILE_FORMAT%label)) as fin_vals:
                length_tgt = len(fin_tgt.read().split("\n"))
                length_vals = len(fin_vals.read().split("\n"))
                assert length_tgt == length_vals
