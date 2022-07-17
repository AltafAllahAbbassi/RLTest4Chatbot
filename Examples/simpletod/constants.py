from RLTest4Chatbot.constants import PROJECT_PATH

# Simple_Tod(S_TOD) Configuration
S_TOD_PATH = PROJECT_PATH + "Examples/simpletod/simpletod/"
S_TOD_MODEL_CHECKPOINT = S_TOD_PATH + "baseline/checkpoint-825000"
S_TOD_EOT_TOKEN = '<|endoftext|>'  # end_of_text
S_TOD_EOB_TOKEN = '<|endofbelief|>'  # end_of_belief
S_TOD_EOC_TOKEN = '<|endofcontext|>'  # end_of_context
S_TOD_BELIEF_TOKEN = '<|belief|>'
S_TOD_CONTEXT_TOKEN = '<|context|>'
S_TOD_USER_TOKEN = '<|user|>'
S_TOD_SYSTEM_TOKEN = '<|system|>'
S_TOD_IGNORE_NONE_DONT_CARE = True  # ignore none and don't care
S_TOD_DEFAULT_CLEANING = True
S_TOD_TYPE2_CLEANING = False
S_TOD_MAPPING_PAIR = S_TOD_PATH + "utils/multiwoz/mapping.pair"