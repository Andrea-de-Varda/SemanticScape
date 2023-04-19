import os
from tqdm import tqdm
import re
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import pickle
import numpy as np
import pandas as pd
from itertools import combinations
import statsmodels.api as sm
from researchpy import corr_pair
import rsatoolbox
from glob import glob

os.chdir("/path/to/your/wd")
    
with open("vizgen_rank_25D", 'rb') as handle:
    d_25d = pickle.load(handle)
    
def cosine(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

def cos_from_d(thedict, w1, w2):
    out = np.nan
    try:
        out = cosine(thedict[w1], thedict[w2])
    except KeyError:
        pass
    return out


#######################################
# loading standard text-based vectors #
#######################################

filesem = [item.split() for item in open("/path/to/your/SGNS-w10-d300.decoded.txt", encoding="utf8").readlines()] # word2vec model from Lenci, 2022. Link in the README.
semvec = {}
for item in tqdm(filesem):
    try:
        vec = np.array([float(num) for num in item[1:]])
        semvec[item[0]] = vec
    except ValueError:
        pass
semvec = {key:value for key, value in semvec.items() if value.shape == (300,)}

#############################
# loading CNN-based vectors #
#############################

vispa = [item.split() for item in open("path/to/your/ViSpa_prototypes.txt", encoding="utf8").readlines()] # ViSpa representations from Gunther. Link in the README.
vsp = {}
for item in tqdm(vispa):
    vec = np.array([float(num) for num in item[1:]])
    if vec.shape == (400,):
        vsp[item[0][1:-5].lower()] = vec
print(len(vsp)) # 7801

#######
# RSA #
#######

def normalize(x):
    return x / np.linalg.norm(x)

of_interest = set.intersection(set(semvec.keys()), set(d_25d.keys()), set(vsp.keys()) )
len(of_interest) # 3172

A = np.array([semvec[w] for w in of_interest])
B = np.array([vsp[w] for w in of_interest])
C = np.array([d_25d[w] for w in of_interest])

# Cosine distance not implemented in rsatoolbox. Normalize, then euclidean (same)

A = np.array([normalize(semvec[w]) for w in of_interest]) # Semantic
B = np.array([normalize(vsp[w]) for w in of_interest]) # ViSpa
C = np.array([normalize(d_25d[w]) for w in of_interest]) # SSM

A = rsatoolbox.data.Dataset(A)
rdm_a = rsatoolbox.rdm.calc_rdm(A)

B = rsatoolbox.data.Dataset(B)
rdm_b = rsatoolbox.rdm.calc_rdm(B)

C = rsatoolbox.data.Dataset(C)
rdm_c = rsatoolbox.rdm.calc_rdm(C)

rsatoolbox.rdm.compare_rho_a(rdm_a, rdm_b) # 0.21112215503955895, semantic-vispa
rsatoolbox.rdm.compare_rho_a(rdm_a, rdm_c) # 0.1483373743937731, semantic-semscape
rsatoolbox.rdm.compare_rho_a(rdm_b, rdm_c) # 0.16645778225748695, vispa-semscape

#############################
# predict visual similarity #
#############################

colnames = ["sem", "vispa", "sim_25"]

img_sim = pd.read_csv("/path/to/your/ViSpa_prototype_visual_sim.csv", sep=" ")

sem_vz = []; vispa_vz = []; sim_25_vz = []
for index, row in img_sim.iterrows():
    sem_vz.append(cos_from_d(semvec, row.word1, row.word2))
    vispa_vz.append(cos_from_d(vsp, row.word1, row.word2))
    sim_25_vz.append(cos_from_d(d_25d, row.word1, row.word2))

colnames_vz = [item+"_vz" for item in colnames]
for thelist, thename in zip([sem_vz, vispa_vz, sim_25_vz], colnames_vz):
    img_sim[thename] = thelist
    
img_sim_all = img_sim.dropna()
all_corr_vz = corr_pair(img_sim[["Value"]+colnames_vz])
corr_pair(img_sim_all[["sim_25_vz", "Value", "vispa_vz", "sem_vz"]])

x = img_sim_all[["sem_vz", "vispa_vz", "sim_25_vz"]]
y = img_sim_all["Value"]

x = sm.add_constant(x) # adding a constant
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
print_model = model.summary()
print(print_model)

###################################
# ViSpa similarity between images #
###################################

img_sim_img = pd.read_csv("/path/to/your/ViSpa/study2_ratings_images_scores.txt", sep=" ")

sem_vz = []; vispa_vz = []; sim_25_vz = []
for index, row in img_sim_img.iterrows():
    sem_vz.append(cos_from_d(semvec, row.word1, row.word2))
    vispa_vz.append(cos_from_d(vsp, row.word1, row.word2))
    sim_25_vz.append(cos_from_d(d_25d, row.word1, row.word2))

colnames_vz = [item+"_vz" for item in colnames]
for thelist, thename in zip([sem_vz, vispa_vz, sim_25_vz], colnames_vz):
    img_sim_img[thename] = thelist
    
img_sim_all_img = img_sim_img.dropna()
# all_corr_vz = corr_pair(img_sim_img[["Value"]+colnames_vz])
# corr_pair(img_sim_all_img[["sim_25_vz", "Value", "vispa_vz", "sem_vz"]])

x = img_sim_all_img[["sem_vz", "vispa_vz", "sim_25_vz"]]
y = img_sim_all_img["Value"]

x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
print_model = model.summary()
print(print_model)

#####################################
# thematic vs taxonomic relatedness #
#####################################

##########################
# taxonomic associations #
##########################

simlex = [line.split("\t") for line in open("path/to/your/SimLex-999/SimLex-999.txt").readlines()]
simlex = pd.DataFrame(simlex[1:], columns=simlex[0])
simlex["SimLex999"] = pd.to_numeric(simlex["SimLex999"])
simlex["sem"] = [cos_from_d(semvec, row.word1, row.word2) for index, row in simlex.iterrows()]
simlex["vispa"] = [cos_from_d(vsp, row.word1, row.word2) for index, row in simlex.iterrows()]
simlex["sim_25"] = [cos_from_d(d_25d, row.word1, row.word2) for index, row in simlex.iterrows()]

simlex_all = simlex.dropna()
corr_pair(simlex_all[["SimLex999", "sem", "vispa", "sim_25"]])

x = simlex_all[["sem", "vispa", "sim_25"]]
y = simlex_all["SimLex999"]

x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
print_model = model.summary()
print(print_model)

#########################
# thematic associations # (see http://wordnet.cs.princeton.edu/downloads.html)
#########################

them_labels = open("/path/to/your/ThematicAssociation/controlled.synsets").readlines()
them_pairs = []
for line in them_labels:
    w1, w2 = line.split(" ")
    w1 = re.sub(r"(.*?)(%)(.*)", r"\1", w1)
    w2 = re.sub(r"(.*?)(%)(.*)", r"\1", w2)
    w2 = re.sub("\n", "", w2)
    them_pairs.append([w1, w2])
    
them_values = open("/path/to/your/ThematicAssociation/controlled.standard").readlines()
them_scores = []
for line in them_values:
    line = [float(n) for n in re.sub("\n", "", line).split(" ") if n]
    them_scores.append(np.mean(line))

them_df = pd.DataFrame(them_pairs, columns=["w1", "w2"])
them_df["score"] = them_scores
them_df["sem"] = [cos_from_d(semvec, row.w1, row.w2) for index, row in them_df.iterrows()]
them_df["vispa"] = [cos_from_d(vsp, row.w1, row.w2) for index, row in them_df.iterrows()]
them_df["sim_25"] = [cos_from_d(d_25d, row.w1, row.w2) for index, row in them_df.iterrows()]

them_df_all = them_df.dropna()
corr_pair(them_df_all[["score", "sem", "vispa", "sim_25"]])

x = them_df_all[["sem", "vispa", "sim_25"]]
y = them_df_all["score"]

x = sm.add_constant(x) # adding a constant
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
print_model = model.summary()
print(print_model)


########################
# image discrimination # # non sig, but neither w2v is
########################

img_discrimination = pd.read_csv("/path/to/your/ViSpa/study4_discrimination_itemlevel.csv", sep=",")

word1, word2 = [], []
for index, row in img_discrimination.iterrows():
    tx = row.pair.lower()
    tx = re.sub("_ntp", "", tx)
    w1, w2 = tx.split("___")
    word1.append(w1); word2.append(w2)
    
img_discrimination["word1"] = word1
img_discrimination["word2"] = word2

sem_vz = []; vispa_vz = []; sim_25_vz = []
for index, row in img_discrimination.iterrows():
    sem_vz.append(cos_from_d(semvec, row.word1, row.word2))
    vispa_vz.append(cos_from_d(vsp, row.word1, row.word2))
    sim_25_vz.append(cos_from_d(d_25d, row.word1, row.word2))

colnames_vz = [item+"_vz" for item in colnames]
for thelist, thename in zip([sem_vz, vispa_vz, sim_25_vz], colnames_vz):
    img_discrimination[thename] = thelist
    
img_discrimination_all = img_discrimination.dropna()

x = img_discrimination_all[["sem_vz", "vispa_vz", "sim_25_vz"]]
y = img_discrimination_all["rt"]

x = sm.add_constant(x) # adding a constant
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
print_model = model.summary()
print(print_model)

#################
# image priming # # non sig
#################

img_priming = pd.read_csv("/path/to/your/ViSpa/study5_priming_itemlevel.csv", sep=",")

word1, word2 = [], []
for index, row in img_priming.iterrows():
    tx = row.pair.lower()
    tx = re.sub("_ntp", "", tx)
    w1, w2 = tx.split("___")
    word1.append(w1); word2.append(w2)
    
img_priming["word1"] = word1
img_priming["word2"] = word2

sem_vz = []; vispa_vz = []; sim_25_vz = []
for index, row in img_priming.iterrows():
    sem_vz.append(cos_from_d(semvec, row.word1, row.word2))
    vizgen_pmi_vz.append(cos_from_d(d_vg_PPMI, row.word1, row.word2))
    vispa_vz.append(cos_from_d(vsp, row.word1, row.word2))
    sim_25_vz.append(cos_from_d(d_25d, row.word1, row.word2))

colnames_vz = [item+"_vz" for item in colnames]
for thelist, thename in zip([sem_vz, vispa_vz, sim_25_vz], colnames_vz):
    img_priming[thename] = thelist
    
img_priming_all = img_priming.dropna()

x = img_priming_all[["sem_vz", "vispa_vz", "sim_25_vz"]]
y = img_priming_all["rt"]

x = sm.add_constant(x) # adding a constant
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
print_model = model.summary()
print(print_model)

####################
# semantic priming #
####################

priming = pd.read_csv("/path/to/your/semantic priming/Database.csv")
priming["LingSim"] = [cos_from_d(semvec, row.prime.lower(), row.target.lower()) for index, row in priming.iterrows()]
priming["VisualSimilarity"] = [cos_from_d(vsp, row.prime.lower(), row.target.lower()) for index, row in priming.iterrows()]
priming["prime"] = priming["prime"].str.lower()
priming["target"] = priming["target"].str.lower()
priming["sim_25"] = [cos_from_d(d_25d, row.prime, row.target) for index, row in priming.iterrows()]

priming = priming.dropna(subset=["sim_25", "VisualSimilarity", "LingSim"])
print(priming.columns)

x_visual_25 = sm.add_constant(priming[["sim_25", "VisualSimilarity", "LingSim", "LgSUBTLWF.prime", "LgSUBTLWF.target", "Length.prime", "Length.target", "OLD.prime", "OLD.target", "PLD.prime", "PLD.target", "NSyll.prime", "NSyll.target", "LgSUBTLCD.prime", "LgSUBTLCD.target"]])

out = []
for y_measure in ["NT_200ms_RT", "NT_1200ms_RT", "LDT_200ms_RT", "LDT_1200ms_RT"]:
    print(y_measure)
    y_priming = np.log(priming[y_measure])
    model_visual_25 = sm.OLS(y_priming, x_visual_25).fit(); print(model_visual_25.summary())
    t_lang = model_visual_25.tvalues["LingSim"]#model_language.tvalues["LingSim"]
    t_vispa = model_visual_25.tvalues["VisualSimilarity"]#model_visual.tvalues["VisualSimilarity"]
    t_v25 = model_visual_25.tvalues["sim_25"]
    out.append([y_measure, t_lang, t_vispa, t_v25])
out = pd.DataFrame(out, columns=["measure", "lang", "vispa", "sim_25"])
print(out)



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

encyclopedic_semantics = glob("path/to/your/BATS_3.0/3_Encyclopedic_semantics/*")
e_s = {re.sub("(.*)(\[.*\])(\.txt)", r"\2", line)[1:-1] : analogy_expander(line) for line in encyclopedic_semantics}

lexicographic_semantics = glob("path/to/your/BATS_3.0/4_Lexicographic_semantics/*")
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
            for word in set.difference(of_interest, set([a, b, c])): # constraint the search space without a, b, c
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
        for word in set.difference(of_interest, set([a, b, c])):
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
    McNemar(w2v_vsp_acc, w2v_acc)
    
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
    McNemar(all_acc, w2v_vsp_acc)
    
    
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
