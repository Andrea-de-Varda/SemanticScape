import glob
import os
from tqdm import tqdm
import re
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import pickle
import numpy as np
from researchpy import ttest
import pandas as pd
from itertools import combinations
import statsmodels.api as sm
from researchpy import corr_pair
from itertools import combinations
from glob import glob

os.chdir("/path/to/your/wd")
    
with open("vizgen_rank_25D", 'rb') as handle:
    d_25d = pickle.load(handle)
    
def cosine(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

def cos_from_d(d, key1, key2):
    try:
        cos = cosine(d[key1], d[key2])
    except KeyError:
        cos = np.nan
    return cos

#######################################
# loading standard text-based vectors #
#######################################

filesem = [item.split() for item in open("/path/to/your/SGNS-w10-d300.decoded.txt", encoding="utf8").readlines()]
semvec = {}
for item in tqdm(filesem):
    try:
        vec = np.array([float(num) for num in item[1:]])
        semvec[item[0]] = vec
    # if vec.shape == (100,):
    #     semvec[item[0]] = vec
    except ValueError:
        pass
semvec = {key:value for key, value in semvec.items() if value.shape == (300,)}

#############################
# loading CNN-based vectors #
#############################

vispa = [item.split() for item in open("path/to/your/ViSpa_prototypes.txt", encoding="utf8").readlines()]
vsp = {}
for item in tqdm(vispa):
    vec = np.array([float(num) for num in item[1:]])
    if vec.shape == (400,):
        vsp[item[0][1:-5].lower()] = vec
print(len(vsp)) # 7801

###########
# ANALOGY #
###########
# test if rank of target word increases combining different sources of information (derive three rankings with respect to prediction)

# frame it as a logistic regression problem with unbalanced classes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

of_interest = set.intersection(set(semvec.keys()), set(d_25d.keys()), set(vsp.keys())) 

def analogy_expander(location):
    analogy = open(location).readlines()
    analogy =[re.sub("\n", "", w).split("\t") for w in analogy]
    temp = []
    for num1, num2 in analogy:
        num2 = num2.split("/")
        for word in num2:
            if len(set.intersection(set([num1]), of_interest)) == len(set.intersection(set([word]), of_interest)) == 1:
                temp.append([num1, word])
    analogies = list(combinations(temp, 2))
    return analogies

encyclopedic_semantics = glob("/path/to/your/BATS_3.0/3_Encyclopedic_semantics/*")
e_s = {re.sub("(.*)(\[.*\])(\.txt)", r"\2", line)[1:-1] : analogy_expander(line) for line in encyclopedic_semantics}

lexicographic_semantics = glob("/path/to/your/BATS_3.0/4_Lexicographic_semantics/*")
l_s = {re.sub("(.*)(\[.*\])(\.txt)", r"\2", line)[1:-1] : analogy_expander(line) for line in lexicographic_semantics}

print({key: len(value) for key, value in e_s.items()})
print({key: len(value) for key, value in l_s.items()})



# a : b = c : ?
# argmax(sim(t, c+b-a)

def Sort(sub_li):
    sub_li.sort(key = lambda x: x[1], reverse=True)
    return sub_li

def analogy_solver(analog_list, d, downsample=False):
    analogy_out = []
    if downsample:
        analog_list = analog_list[:10000]
    for item in tqdm(analog_list):
        a, b = item[0]
        c, actual_word = item[1]
        #print(item[0])
        if len(set([a, b, c, actual_word])) == 4: # no repetitions
            t = d[c] + d[b] - d[a]
            out = []
            for word in sorted(set.difference(of_interest, set([a, b, c]))): # constraint the search space without a, b, c
                vec = d[word]
                out.append([word, cosine(vec, t)])
            out = [x[0] for x in Sort(out)]
            analogy_out.append([item, {theword : idx for idx, theword in enumerate(out)}])
    return analogy_out

def McNemar(acc1, acc2):
    acc = zip(acc1, acc2)
    a, b, c, d = 0, 0, 0, 0
    for z in acc:
        if z == (1, 1):
            a += 1
        if z == (1, 0):
            b += 1
        if z == (0, 1):
            c += 1
        if z == (0, 0):
            d += 1
    table = [[a, b], [c, d]]
    #print(table, "\n")
    if min([a, b, c, d]) < 25: 
        result = mcnemar(table, exact=True)
        print("EXACT TEST (< 25)\n")
    else:
        result = mcnemar(table, exact=False, correction=True)
        print("Standard calculation (> 25)\n")
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    return result.statistic, result.pvalue

def test_analogy_logistic(condition_name, analogy_dict=e_s, downsample=False):
    
    print(condition_name.upper(), "\n")

    dvm_an = analogy_solver(analogy_dict[condition_name], d_25d, downsample=downsample)
    vispa_an = analogy_solver(analogy_dict[condition_name], vsp, downsample=downsample)
    sem_an  = analogy_solver(analogy_dict[condition_name], semvec, downsample=downsample)
    #print("Check =", len(dvm_an) == len(vispa_an) == len(sem_an)) # check
    
    out_analogy = []
    for item_sem, item_vispa, item_dvm in tqdm(zip(sem_an, vispa_an, dvm_an), total=len(sem_an)):
        a, b = item_sem[0][0]
        c, actual_word = item_sem[0][1]
        for word in sorted(set.difference(of_interest, set([a, b, c]))):
            rank_sem = item_sem[1][word]
            rank_vispa = item_vispa[1][word]
            rank_dvm = item_dvm[1][word]
            if word == actual_word:
                out_analogy.append([1, rank_sem, rank_vispa, rank_dvm])
            else:
                out_analogy.append([0, rank_sem, rank_vispa, rank_dvm])
    out_analogy = pd.DataFrame(out_analogy, columns = ["class", "sem", "vispa", "dvm"])
    
    #print(out_analogy[out_analogy["class"] == 1].mean())
    #print(out_analogy[out_analogy["class"] == 0].mean())
    print("N =", len(out_analogy))
    
    print("\n\n Word2Vec")
    x = out_analogy.drop(['class', 'vispa', 'dvm'],axis=1)
    y = out_analogy['class']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13)
    lg2 = LogisticRegression(random_state=13, class_weight="balanced")
    lg2.fit(X_train,y_train)
    y_pred = lg2.predict(X_test)
    w2v_acc = list((y_test == y_pred).astype(int))
    print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
    print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
    print(f'Recall score: {recall_score(y_test,y_pred)}\n')
    print(f"\n\nTable for LaTeX \n{accuracy_score(y_test,y_pred)} & {recall_score(y_test,y_pred)}")
    
    print("\n\n Word2Vec + ViSpa")
    x = out_analogy.drop(['class', 'dvm'],axis=1)
    y = out_analogy['class']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13)
    lg2 = LogisticRegression(random_state=13, class_weight="balanced")
    lg2.fit(X_train,y_train)
    y_pred = lg2.predict(X_test)
    w2v_vsp_acc = list((y_test == y_pred).astype(int))
    print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
    print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
    print(f'Recall score: {recall_score(y_test,y_pred)}\n')
    
    print("\nMcNemar - (Word2Vec) - (Word2Vec + Vispa)")
    chi, p = McNemar(w2v_vsp_acc, w2v_acc)
    print(f"\n\nTable for LaTeX \n{accuracy_score(y_test,y_pred)} & {recall_score(y_test,y_pred)} & {chi} & {p}")
    
    print("\n\n Word2Vec + ViSpa + DVM")
    x = out_analogy.drop(['class'],axis=1)
    y = out_analogy['class']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13)
    lg2 = LogisticRegression(random_state=13, class_weight="balanced")
    lg2.fit(X_train,y_train)
    y_pred = lg2.predict(X_test)
    all_acc = list((y_test == y_pred).astype(int))
    print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
    print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
    print(f'Recall score: {recall_score(y_test,y_pred)}')
    
    print("\nMcNemar - (Word2Vec + Vispa) - (ALL)")
    
    chi, p = McNemar(all_acc, w2v_vsp_acc)
    print(f"\n\nTable for LaTeX \n{accuracy_score(y_test,y_pred)} & {recall_score(y_test,y_pred)} & {chi} & {p}")
    
    
# note on confusion matrix: 
    # [true negatives, false negatives,
    #  false positives, true positives]
# (min 500 occurrences)

# Encyclopedic semantics
test_analogy_logistic(condition_name="animal - shelter")
test_analogy_logistic(condition_name="things - color")
test_analogy_logistic(condition_name="animal - young")

# Lexicographic semantics
test_analogy_logistic(condition_name="meronyms - substance", analogy_dict=l_s)
test_analogy_logistic(condition_name="hyponyms - misc", analogy_dict=l_s, downsample=True)
test_analogy_logistic(condition_name="hypernyms - animals", analogy_dict=l_s)
test_analogy_logistic(condition_name="hypernyms - misc", analogy_dict=l_s)
test_analogy_logistic(condition_name="meronyms - part", analogy_dict=l_s, downsample=True)
