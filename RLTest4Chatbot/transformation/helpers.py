import nltk
from nltk.corpus import wordnet 
import numpy as np
import math
from RLTest4Chatbot.transformation.constants import WORD_TAG_DICT
from collections import defaultdict




def char_repeat(sentence, char_index):
    char = sentence[char_index]
    sentence = sentence[:char_index] + char + sentence[char_index:]
    return sentence


def char_insert(sentence, char_index, char):
    sentence = sentence[:char_index] + char + sentence[char_index:]
    return sentence


def get_char(number, min=ord('a'), max=ord('z')):
    return round((max-min)*number + min)


def char_drop(sentence, char_index):
    sentence = sentence[:char_index] + sentence[char_index+1:]
    return sentence


def char_replace(sentence, char_index, char):
    sentence = sentence[:char_index] + char + sentence[char_index+1:]
    return sentence


def char_swap(sentence, char_index):
    """
    Swap between chars at char_index and char_index+1
    If char_index is the last element swap between chars at char_index-1 and at char_index
    """
    length_ = len(sentence)
    if char_index == length_-1:  # the last element in the sentence
        sentence = sentence[:char_index-1] + \
            sentence[char_index] + sentence[char_index-1]
    else:
        sentence = sentence[:char_index] + sentence[char_index +
                                                    1] + sentence[char_index] + sentence[char_index+2:]
    return sentence


def word_insert(sentence, word_index, word):
    words = nltk.word_tokenize(sentence)
    words[word_index] = words[word_index] + " " + word
    return " ".join(words)


def word_piece_insert(sentence, word_index, model, tokenizer):
    """
    Insert a word given by bert pre-trained model
    PS : It's not the same model used in Dial test (uses masked whole word)
    """
    input_ids = tokenizer(
        sentence, return_tensors="pt").input_ids.tolist()[0][1:-1]
    input_ids.insert(word_index, 103)  # 103 is the [MASK] token
    sentence = tokenizer.decode(input_ids)
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    mask_token_indexs = [i for i in range(len(input_ids)) if(
        input_ids[i] == tokenizer.mask_token_id)]
    predicted_token_id = [logits[0, mask_token_index].argmax(
        axis=-1).item() for mask_token_index in mask_token_indexs]
    for i in range(len(mask_token_indexs)):
        index = mask_token_indexs[i]
        input_ids[index] = predicted_token_id[i]
    senetnce = tokenizer.decode(input_ids)
    return senetnce


def word_drop(sentence, word_index):
    words = nltk.word_tokenize(sentence)
    if len(words) == 0:
        return sentence
    word_index = min(word_index, len(words)-1)
    words[word_index] = ""
    return " ".join(words)


def word_replace(sentence, word_index, word):
    words = nltk.word_tokenize(sentence)
    words[word_index] = word
    return " ".join(words)


def word_swap(sentence, word_index):
    words = nltk.word_tokenize(sentence)
    if word_index == len(words) - 1:  # last word
        swap_index = word_index-1
    else:
        swap_index = word_index+1

    swap = words[swap_index]
    word = words[word_index]

    words[word_index] = swap
    words[swap_index] = word
    return " ".join(words)


def get_general_tag(tag: str):
    """
    Given a specific word tag, return the genral
    """
    tags = ["verb", "adjectif", "adverb", "noun"]
    for t in tags:
        if tag in WORD_TAG_DICT[t]:
            return t
    return None


def get_synonyms(word: str, word_tag: str):
    """
    Given a word and a tag, returns a list of the word synonyms having the same genral tag
    """
    word_tag = get_general_tag(word_tag)
    if word_tag == None:
        return [word]

    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.add(l.name())
    if word in synonyms:
        synonyms.remove(word)
    if len(synonyms) == 0:
        return [word]
    syn_list_ = [synonym.replace("-", " ").replace("_", " ")
                 for synonym in list(synonyms)]
    syn_tags_list = nltk.pos_tag(syn_list_)
    syn_list = []
    for (syn, tag) in syn_tags_list:
        if tag in WORD_TAG_DICT[word_tag]:
            syn_list.append(syn)
    if len(syn_list) == 0:
        return [word]
    return syn_list


def construct_dict_file(file: str):
    """
    Given a file, that contains words and corresponding misspelled,
    returns a dictionnary {"word" : [misspelled words]}
    """
    punkt = '!#$%&()*+,-./:;<=>?@[\]^_{|}~'
    file = open(file, 'r')
    lines = file.readlines()
    misspelled = dict()
    for line in lines:
        line = line.lower()
        line = line.strip()
        line = line.translate(str.maketrans('', '', punkt))
        words = line.split(" ")
        key = words[0]
        value = words[1:]
        if key not in list(misspelled.keys()):
            misspelled[key] = value
        else:
            misspelled[key].extend(value)
    return misspelled


def get_active_params(params, rate, vector_size):
    hidden_trans = len(params) // vector_size
    active_trans = round(rate * (hidden_trans))
    active_params = params[:vector_size*active_trans-1]
    return active_params


def calculate_modif_rate(sentence, sentence_): # word swap is not fully explored but it is okey
    """
    Jaccad distance
    """
    words = nltk.word_tokenize(sentence)
    words_ =  nltk.word_tokenize(sentence_)
    if len(words_)==0 or len(words) == 0 :
        return 1
    shared = set(words).intersection(words_)
    n_shared = len(shared)
    union = set(words).union(words_)
    n_union = len(union)
    return 1- n_shared/n_union




"""
Most of the code from https://github.com/alvations/pywsd
Paper : https://arxiv.org/pdf/1802.05667.pdf
"""

alpha = 0.2
beta  = 0.45
benchmark_similarity = 0.8025
gamma = 1.8


def _disambiguate(sentence):
    wsd =[]
    words = nltk.word_tokenize(sentence)
    for word in words : 
        try :
            xx = wordnet.synsets(word)[1]
        except :
            xx = None
        wsd.append((word,xx))
    return wsd

def calculate_similarity_sen(sentence1, sentence2):
    L1 =dict()
    L2 =defaultdict(list)
    s1_wsd = _disambiguate(sentence1)
    s2_wsd = _disambiguate(sentence2)
    s1 = [syn  for syn in s1_wsd if syn[1]] # not None
    s2 = [syn  for syn in s2_wsd if syn[1]] # not None
    for syn1 in s1:
        L1[syn1[0]] =list()
        for syn2 in s2:                                     
            
            subsumer = syn1[1].lowest_common_hypernyms(syn2[1], simulate_root=True)[0]
            h =subsumer.max_depth() + 1 # as done on NLTK wordnet        
            syn1_dist_subsumer = syn1[1].shortest_path_distance(subsumer,simulate_root =True)
            syn2_dist_subsumer = syn2[1].shortest_path_distance(subsumer,simulate_root =True)
            l  =syn1_dist_subsumer + syn2_dist_subsumer
            f1 = np.exp(-alpha*l)
            a  = np.exp(beta*h)
            b  = np.exp(-beta*h)
            f2 = (a-b) /(a+b)
            sim = f1*f2
            L1[syn1[0]].append(sim)          
            L2[syn2[0]].append(sim)
    V1 =np.array( [max(L1[key]) for key in L1.keys()])
    V2 = np.array([max(L2[key]) for key in L2.keys()])
    S  = np.linalg.norm(V1)*np.linalg.norm(V2)
    C1 = sum(V1>=benchmark_similarity)
    C2 = sum(V2>=benchmark_similarity)
    Xi = (C1+C2) / gamma
    if C1+C2 == 0:
            Xi = max(V1.size, V2.size) / 2

    return S/Xi

def modif_rate_sen(sentence1, sentence2):
    try :
        modif = 1-calculate_similarity_sen(sentence1, sentence2)
    except : 
        modif = 0
    if modif>1:
        modif =1 
    elif modif <0:
        modif = 0
    return modif

def jaccard_modif_rate(word1, word2):
    word1_l = list(word1)
    word2_l = list(word2)
    shared = set(word1_l).intersection(word2_l)
    if len(shared) == 0:
        return 1
    union = set(word1_l).union(word2_l)

    return 1-len(union)/len(shared)

def nltk_modif(word1, word2):
    try :
        syn1 = wordnet.synsets(word1)[1]
        syn2 = wordnet.synsets(word2)[1]
        sim = wordnet.wup_similarity(syn1, syn2)
        
    except :
        sim = 0

    return 1-sim
    


def jaro_distance__(s1, s2, mm= 2):
    if (s1 == s2):
        return 0.0
 
    len1 = len(s1)
    len2 = len(s2)

    max_dist = math.floor(max(len1, len2) / mm) - 1
    match = 0
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
    for i in range(len1):
        for j in range(max(0, i - max_dist),
                       min(len2, i + max_dist + 1)):             
            if (s1[i] == s2[j] and hash_s2[j] == 0):
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break
 
    if (match == 0):
        return 0.0
    t = 0
    point = 0
    for i in range(len1):
        if (hash_s1[i]):
            while (hash_s2[point] == 0):
                point += 1
 
            if (s1[i] != s2[point]):
                t += 1
            point += 1
    t = t//2
    
    return 1-(match/ len1 + match / len2 +
            (match - t) / match)/ 3.0


