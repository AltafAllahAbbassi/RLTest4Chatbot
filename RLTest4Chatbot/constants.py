import uuid


ENVIRONMENT = 'cpu'
PROJECT_PATH = './'
FRAMEWORK_PATH = PROJECT_PATH +"RLTest4Chatbot/"
BASELINE_PATH = PROJECT_PATH + "DialTest/"


# back translation configuration
BT_KEY = ""
BT_ENDPOINT = ""
BT_LOCATION = "francecentral"
BT_PATH = '/translate'
BT_PARAMS = {
    'api-version': '3.0',
    'to': []
}

BT_HEADERS = {
    'Ocp-Apim-Subscription-Key': BT_KEY,
    'Ocp-Apim-Subscription-Region': BT_LOCATION,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

BT_BODY = [{"text": ''}]


# Full Form to contracted form https://en.wikipedia.org/wiki/Contraction_(grammar)#English
FULL_FORM2CONTRACTED_FORM = {
    " not ": "n't ",  # I can not => I cann't
    "let us": "let's",
    "I am":	"I'm",
    " are": "'re",
    " does": "'s",
    " is": "'s",
    " has": "'s",
    " have": "'ve",
    " had": "'d",
    " did": "'d",
    " would": "'d",
    " will": "'ll",
    " shall": "'ll",
    "of ": "o'",
    "of the ": "o'",
    "it was":	"'twas",
    "them":	"'em",
    "you ": "y'",
    "about": "'bout",
    "because":	"'cause"
}

# Contracted Form to Full Form  https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
CONTRACTED_FORM2FULL_FORM = {
}

WWI_MODEL_PATH = BASELINE_PATH +  "wwm_cased_L-24_H-1024_A-16/"

MULTIWOZ_21_PATH = PROJECT_PATH + "Examples/MultiWOZ/data/MultiWOZ_2.1/"
MULTIWOZ_22_PATH = PROJECT_PATH + "Examples/MultiWOZ/data/MultiWOZ_2.2/MultiWOZ2.1Format/"