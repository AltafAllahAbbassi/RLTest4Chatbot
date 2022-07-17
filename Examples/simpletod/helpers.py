from Examples.simpletod.constants import S_TOD_DEFAULT_CLEANING,S_TOD_IGNORE_NONE_DONT_CARE, S_TOD_EOB_TOKEN, S_TOD_USER_TOKEN, S_TOD_SYSTEM_TOKEN, S_TOD_BELIEF_TOKEN, S_TOD_CONTEXT_TOKEN, S_TOD_EOC_TOKEN, S_TOD_EOT_TOKEN
from Examples.simpletod.simpletod.utils.multiwoz.nlp import normalize_lexical, normalize_beliefstate
from Examples.simpletod.simpletod.utils.dst import default_cleaning, ignore_none_dontcare, ignore_not_mentioned

def dial2text(dialogue: list, turn_id: int, user_input: str):
    """
    Given a dialogue a turn_id and a user utterance, it converts, all, to the input format of simple TOD model
    Inputs :dialogue => a list of turns
            turn_id => the id of the turn, we are interessted about  
            user_input => the user transcript (response) at turn number = turn_id
    Outputs :the converted dialogue + turn_id+ user_input under this format :
            <|endoftext|> <|context|> <|user|> ... <|system|> .... <|user|> .... <|endofcontext|> 
    """
    for idx, turn in enumerate(dialogue["dialogue"]):
        if (turn_id == 0):
            context = (S_TOD_USER_TOKEN +
                       ' {}'.format(normalize_lexical(user_input)))
            break
        elif (idx < turn_id):
            if (idx == 0):
                context = S_TOD_USER_TOKEN + \
                    ' {}'.format(normalize_lexical(turn["transcript"]))
            else:
                context += (' ' + S_TOD_SYSTEM_TOKEN + ' {}'.format(
                    normalize_lexical(turn["system_transcript"])) + ' ' + S_TOD_USER_TOKEN + ' {}'.format(
                    normalize_lexical(turn["transcript"])))
        else:
            context += (' ' + S_TOD_SYSTEM_TOKEN + ' {}'.format(
                normalize_lexical(turn["system_transcript"])) + ' ' + S_TOD_USER_TOKEN + ' {}'.format(
                normalize_lexical(user_input)))
            break
    text = S_TOD_EOT_TOKEN + ' ' + S_TOD_CONTEXT_TOKEN + \
        ' ' + context + ' ' + S_TOD_EOC_TOKEN
    return text.strip()


def get_belief_new_dbsearch(sent: str):
    """
    Given an intermediate prediction ( =context + predicted belief), extracts the predicted belief 
    Inputs :sent => intermediate prediction under the format <|endoftext|><|context|> <|user|> ... <|system|> ... <|user|> ... <|endofcontext|> <|belief|>....<|endofbelief|>
    Outputs:predicted belief under the format [slot_name_1 value_1, slot_name_2 value_1, ....]
    """
    if S_TOD_BELIEF_TOKEN in sent:
        tmp = sent.strip(' ').split(
            S_TOD_BELIEF_TOKEN)[-1].split(S_TOD_EOB_TOKEN)[0]
    else:
        return []
    tmp = tmp.strip(' .,')
    tmp = tmp.replace(S_TOD_EOB_TOKEN, '')
    tmp = tmp.replace(S_TOD_EOT_TOKEN, '')
    belief = tmp.split(',')
    new_belief = []
    for bs in belief:
        bs = bs.strip(' .,')
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief

def convert_belief(belief: str):
    """
    Given a belief, returns the same belief under a different format
    Input :belief under [slot_name_1 value_1, slot_name_2 value_1, ....]
    Output:belief under {'domain1' : {'slot1 : value1, ...}, ...}
    """
    dic = {}
    for bs in belief:
        if bs in [' ', '']:
            continue
        domain = bs.split(' ')[0]
        slot = bs.split(' ')[1]
        if slot == 'book':
            slot = ' '.join(bs.split(' ')[1:3])
            value = ' '.join(bs.split(' ')[3:])
        else:
            value = ' '.join(bs.split(' ')[2:])
        if domain not in dic:
            dic[domain] = {}
        try:
            dic[domain][slot] = value
        except:
            pass
    return dic

def format_bs(target: list, pred:  dict):
    """
    Given a ground-truth belief, and a predicted belief returns both under a specific format
    Inputs :target => ground-truth belief under the format [{'slots' : [['slot' : value]], 'act' : value}]
           :pred   => predicted belief under the format {'domain1' : {'slot1 : value1, ...}, ...}
    Outputs:turn_target => ground-truth belief under the format ['slot1 value1' , ...] 
           :turn_pred   => predicted belief under the format ['slot1 value1' , ...] 
    """
    turn_pred = []
    turn_target = []
    for slot in target:
        slot, value = slot["slots"][0]
        value = normalize_beliefstate(value)
        slot = slot.replace("-", " ")
        turn_target.append(slot + " " + value)

    for domain in pred:
        for slot, value in pred[domain].items():
            turn_pred.append(domain + " " + slot + " " + value)

    return turn_target, turn_pred

def cal_join_acc(turn_pred: list, turn_target: list):
    """
    Given a ground-truth belief, and a predicted belief returns joint accuracy
    Inputs :turn_target => ground-truth belief  ['slot1 value1' , ...] 
           :turn_pred   => predicted belief ['slot1 value1' , ...] 
    Output :{Prediction : turn_pred,
             Ground Truth : turn_target,
             Joint Acc : 0 or 1
            }
    """
    joint_acc = 0.0
    for bs in turn_pred:
        if bs in [S_TOD_EOT_TOKEN] + ['', ' '] or bs.split()[-1] == 'none':
            turn_pred.remove(bs)
    new_turn_pred = []
    for bs in turn_pred:
        for tok in [S_TOD_EOT_TOKEN]:
            bs = bs.replace(tok, '').strip()
            new_turn_pred.append(bs)
    turn_pred = new_turn_pred

    if S_TOD_DEFAULT_CLEANING:
        turn_pred, turn_target = default_cleaning(turn_pred, turn_target)

    if (S_TOD_IGNORE_NONE_DONT_CARE):
        turn_pred, turn_target = ignore_none_dontcare(turn_pred, turn_target)
    else:
        turn_pred, turn_target = ignore_not_mentioned(
            turn_pred, turn_target)  # Adapted from original result
    if set(turn_target) == set(turn_pred):
        joint_acc = 1.0

    return {"Prediction": sorted(turn_pred),
            "Ground Truth": sorted(turn_target),
            "Joint Acc": joint_acc}