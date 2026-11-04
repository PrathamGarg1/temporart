"""Spec for the 100k+ multiclass thread dataset.

Axes: context length (3/5/7/10 messages) x script (Devanagari Hindi, English,
Hinglish) x class (5, from Van Hee, Guest, Banko, HASOC). Every record carries
subtype, topic and audience so slices are auditable.
"""

from __future__ import annotations

import hashlib
import random
import re

LENGTHS = (3, 5, 7, 10)
SCRIPTS = ("hindi_devanagari", "english", "hinglish")
CLASSES = (
    "benign",
    "friendly_sarcasm",
    "hostile_abuse",
    "sexual_harassment",
    "predatory_grooming",
)
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
CLASS_DIR = {c: f"{i}_{c}" for i, c in enumerate(CLASSES)}

SCRIPT_INSTR = {
    "hindi_devanagari": (
        "Write every message in proper, natural Hindi in Devanagari script only. "
        "No Roman letters and no English words, except unavoidable names."
    ),
    "english": (
        "Write every message in proper, natural everyday English only. "
        "No Hindi or Hinglish words."
    ),
    "hinglish": (
        "Write every message in Hinglish: Hindi in Roman letters mixed with English, "
        "the way Indians chat on WhatsApp. No Devanagari."
    ),
}

NAMES = [
    ("Aarav", "आरव"), ("Riya", "रिया"), ("Kabir", "कबीर"), ("Ananya", "अनन्या"),
    ("Rohan", "रोहन"), ("Priya", "प्रिया"), ("Vikram", "विक्रम"), ("Neha", "नेहा"),
    ("Arjun", "अर्जुन"), ("Sneha", "स्नेहा"), ("Karan", "करण"), ("Pooja", "पूजा"),
    ("Aditya", "आदित्य"), ("Isha", "ईशा"), ("Manav", "मानव"), ("Simran", "सिमरन"),
    ("Rahul", "राहुल"), ("Meera", "मीरा"), ("Yash", "यश"), ("Tanvi", "तन्वी"),
    ("Dev", "देव"), ("Kavya", "काव्या"), ("Nikhil", "निखिल"), ("Divya", "दिव्या"),
    ("Harsh", "हर्ष"), ("Sana", "सना"), ("Imran", "इमरान"), ("Gurleen", "गुरलीन"),
    ("Joseph", "जोसेफ"), ("Lakshmi", "लक्ष्मी"),
]

_FEMALE_ROMAN = {
    "Riya", "Ananya", "Priya", "Neha", "Sneha", "Pooja", "Isha", "Simran", "Meera",
    "Tanvi", "Kavya", "Divya", "Sana", "Gurleen", "Lakshmi",
}
FEMALE = [n for n in NAMES if n[0] in _FEMALE_ROMAN]
MALE = [n for n in NAMES if n[0] not in _FEMALE_ROMAN]

TOPICS = [
    "a cricket match", "the college fest", "exam results", "a group project",
    "a birthday party", "train tickets", "canteen food", "a new phone",
    "a movie night", "the gym", "a traffic jam", "house rent", "an internship",
    "wedding shopping", "festival plans", "the library", "a bike repair",
    "an online class", "hostel mess food", "a weekend trip", "cooking dinner",
    "an office meeting", "a project deadline", "a school picnic", "tuition class",
    "a football game", "a video game ranking", "a music concert", "a pet dog",
    "a laptop problem", "a power cut", "monsoon rain", "the shopping mall",
    "a cab booking", "a birthday gift", "exam notes", "sports day",
    "slow WiFi", "a new haircut", "a school bus",
]

# (subtype, description, audience, weight)
SUBTYPES = {
    "benign": [
        ("logistics", "Friends or colleagues sort out plans, timings and errands.", "peers", 3),
        ("study_help", "Classmates share notes and doubts before an exam.", "peers", 2),
        ("family_chat", "Family members chat about dinner, health or news.", "peers", 2),
        ("parent_child", "A parent talks with a school-age child about homework or pickup.", "minor", 2),
        ("teacher_student", "A teacher politely answers a student's homework doubt.", "minor", 1),
        ("surprise_secret", "Friends harmlessly keep a birthday surprise secret from a friend.", "peers", 1),
        ("consensual_flirt", "Two adults flirt warmly and both are clearly happy about it.", "adult", 1),
        ("work_update", "Colleagues discuss tasks and deadlines.", "peers", 2),
    ],
    "friendly_sarcasm": [
        ("game_roast", "Friends tease each other after a game or match.", "peers", 2),
        ("late_teasing", "Friends tease someone for always being late.", "peers", 1),
        ("mock_praise", "Friends give mock praise for a silly mistake, laughing.", "peers", 2),
        ("exam_banter", "Classmates joke about marks and last-minute studying.", "peers", 1),
        ("affectionate_insult", "Close friends call each other silly names affectionately.", "peers", 2),
        ("fashion_teasing", "Friends tease each other about clothes or haircuts, laughing.", "peers", 1),
    ],
    "hostile_abuse": [
        ("argument", "A heated argument; the last message is a direct hostile insult.", "peers", 3),
        ("breakup_fight", "Two exes fighting; the last message is a cruel personal attack.", "adult", 1),
        ("gaming_toxicity", "Toxic teammates blame each other in an online game.", "peers", 2),
        ("workplace_conflict", "Colleagues in a bitter conflict; the last message is an attack.", "peers", 1),
        ("neighbour_dispute", "Neighbours quarrel; the last message is aggressive abuse.", "peers", 1),
        ("threat", "A non-sexual threat or intimidation of the other person.", "peers", 1),
        ("online_trolling", "A troll attacks a stranger's post in the comments.", "peers", 1),
    ],
    "sexual_harassment": [
        ("boss_employee", "A senior at work makes unwanted suggestive remarks to a junior.", "adult", 2),
        ("classmate_remarks", "A classmate keeps making unwanted suggestive comments.", "adult", 2),
        ("stranger_dm", "A stranger sends unsolicited suggestive messages on a social app.", "adult", 2),
        ("persistent_after_refusal", "Someone keeps pressuring for a date or favours after being refused.", "adult", 2),
        ("coworker_comments", "A coworker comments on appearance in an unwanted, suggestive way.", "adult", 1),
    ],
    "predatory_grooming": [
        ("gaming_stranger", "An adult stranger befriends the child in an online game and offers gifts.", "minor", 2),
        ("social_app_stranger", "An adult pretending to be a teenager befriends the child on a social app.", "minor", 2),
        ("secrecy_push", "An adult tells the child to keep the chats secret from parents.", "minor", 2),
        ("isolation_gifts", "An adult flatters the child, sends gifts and asks when parents are away.", "minor", 2),
        ("move_to_private_app", "An adult asks the child to move to a private app and delete the chat.", "minor", 1),
    ],
}

LAST_RULE = {
    "benign": "a normal, harmless message that fits the conversation.",
    "friendly_sarcasm": (
        "a playful teasing or sarcastic remark that is clearly affectionate; the friends are laughing."
    ),
    "hostile_abuse": (
        "a hostile personal insult or aggressive attack on {Y}, meant to hurt."
    ),
    "sexual_harassment": (
        "an unwanted, suggestive remark or pressure of a sexual nature aimed at {Y}; suggestive but not graphic."
    ),
    "predatory_grooming": (
        "a grooming move toward {Y}: asking to keep the chat secret from parents, to move to a private app, "
        "to meet alone, when the parents are away, or probing {Y}'s age. No sexual content."
    ),
}

CLASS_RULES = {
    "benign": (
        "Nobody is insulted, threatened, sexually pressured, or pushed to hide chats from a parent."
    ),
    "friendly_sarcasm": "The tone stays warm throughout and nobody is truly hurt.",
    "hostile_abuse": (
        "Strong insults and profanity are allowed. No slurs about caste, religion, gender, ethnicity "
        "or disability. No sexual content."
    ),
    "sexual_harassment": (
        "Keep it non-graphic: no explicit sexual words. {Y} deflects or shows discomfort in earlier messages."
    ),
    "predatory_grooming": (
        "Absolutely no sexual content, no talk of bodies, no photo requests. Earlier messages build trust "
        "with games, flattery or gifts."
    ),
}

PAIR_CONTEXT = {
    "friendly_sarcasm": (
        "The earlier messages show friends joking and laughing, so the last line reads as playful teasing."
    ),
    "hostile_abuse": (
        "The earlier messages show a real fight: {Y} is hurt and angry, so the last line reads as a genuine attack."
    ),
}

# Same last line is used for BOTH classes, so only the earlier turns decide the label.
PAIR_BANK = {
    "english": [
        "you are such an idiot", "wow, you really are clueless", "unbelievable, you never learn",
        "what a genius you are", "you're hopeless, honestly", "shut up already", "get a life",
        "you call that a plan?", "you're impossible", "nice going, champ", "you must be joking",
        "are you even listening", "you have no brain", "dumbest thing I've heard",
        "what is wrong with you", "you're a joke", "try using your head", "pathetic, honestly",
        "you're so annoying", "grow up already", "you're useless", "what a loser move",
        "you never get it right", "seriously, are you dumb", "how are you this slow",
        "you're a disaster", "just stop talking", "wow, so smart", "you're out of your mind",
        "that was the worst idea", "give it a rest", "you embarrass me", "how do you even function",
        "you're a clown", "good job, genius", "you are unbelievable",
    ],
    "hindi_devanagari": [
        "तू पागल है क्या", "कितना बेवकूफ है तू", "तेरा दिमाग कहीं और रहता है क्या",
        "वाह, क्या दिमाग लगाया है", "तू कभी नहीं सुधरेगा", "चुप कर यार", "तू तो हद करता है",
        "कुछ अक्ल लगा ले", "क्या बकवास कर रहा है", "तुझसे कुछ नहीं होगा", "कितना बेकार है तू",
        "तू सच में गधा है", "बड़ा आया होशियार", "दिमाग घास चरने गया है क्या", "तू तो जोकर है",
        "कितना धीमा है तू", "तुझे कुछ नहीं आता", "बंद कर ये नाटक", "तू पूरा नालायक है",
        "अक्ल है भी या नहीं", "तू बिल्कुल निकम्मा है", "क्या यार तू भी", "कितना घटिया आइडिया है",
        "तेरे बस की बात नहीं", "बहुत हो गया तेरा", "तू सुधर जा", "तू बहुत परेशान करता है",
        "चल निकल यहाँ से", "वाह रे महान आदमी", "तेरी अक्ल पर तरस आता है",
        "तू तो कमाल का नासमझ है", "तू फालतू है", "क्या पागलपन है ये", "चल झूठे",
        "तेरा दिमाग खराब है", "हद है तेरी तो",
    ],
    "hinglish": [
        "you r an idiot", "tu pagal hai kya", "kitna bewakoof hai tu", "kya dimag lagaya hai, waah",
        "tu kabhi nahi sudhrega", "chup kar yaar", "hadd karta hai tu", "thoda dimag laga le",
        "kya bakwas kar raha hai", "tujhse kuch nahi hoga", "kitna bekaar hai tu",
        "tu sach mein gadha hai", "bada aaya hoshiyar", "dimag ghaas charne gaya hai kya",
        "tu toh joker hai", "kitna slow hai tu", "tujhe kuch nahi aata", "band kar ye natak",
        "tu poora nalayak hai", "akal hai ya nahi", "tu bilkul nikamma hai", "kya yaar tu bhi",
        "kitna ghatiya idea hai", "tere bas ki baat nahi", "bahut ho gaya tera", "tu sudhar ja",
        "bahut pareshan karta hai tu", "chal nikal yahan se", "waah re mahan aadmi",
        "teri akal pe taras aata hai", "tu toh kamaal ka nasamajh hai", "stop acting smart",
        "tu gone case hai", "kya pagalpan hai ye", "chal jhoota", "dimag kharab hai tera",
    ],
}

HIN_MARKERS = set(
    "hai hain nahi nahin kya yaar bhai mujhe tum tu tera teri mera meri ko ka ki ke se mein main "
    "hum aap abhi kal aaj toh bhi par lekin kyun kaise kuch bahut accha acha theek haan chal karo "
    "kar kiya raha rahi ho hoon tha thi wala wali bol bolo sab ab phir jao aao dena lena pata "
    "matlab bas arre arey oye karna hoga hua gaya".split()
)
HIN_STRICT = set(
    "hai hain nahi nahin kya yaar bhai mujhe tum tera teri mera meri mein hum aap abhi kal aaj toh "
    "bhi kyun kaise kuch bahut accha theek haan raha rahi hoon tha thi wala bolo pata matlab bas arre".split()
)

DEV_RE = re.compile(r"[\u0900-\u097F]")
LAT_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[A-Za-z']+")
REFUSAL_RE = re.compile(
    r"(?i)(i (can'?t|cannot|won'?t|am unable)|as an ai|i'?m sorry|cannot (assist|help|comply)|"
    r"against (my|the) (policy|guidelines)|मैं (इस|ऐसा).{0,20}नहीं कर सकत)"
)
EXPLICIT_RE = re.compile(
    r"(?i)\b(fuck|dick|pussy|blowjob|porn|nude|naked|nudes|cock|boobs|breasts?|sex|sexy)\b|"
    r"सेक्स|नंगा|नग्न|लंड|चूत|गांड|nanga"
)
PROFANE_RE = re.compile(
    r"(?i)\b(chutiy\w*|chutya|chut|choot\w*|madarchod|maderchod|mc|bc|bhosd\w*|randi|rundi|"
    r"harami|haraami|gaand|gand|lund|lavd\w*|saala|saale|kutta|kutte|kamina|kamine|bastard|"
    r"bitch|asshole|shit|fuck\w*|slut|whore|dick)\b|"
    r"चूतिय|चुत|मादरचोद|भोसड|रंडी|हरामी|गांड|लंड|साला|साले|कुत्ता|कुत्ते|कमीना|कमीने"
)
GROOM_BAD_RE = re.compile(
    r"(?i)\b(body|kiss|bra|panty|photo|pic|pics|selfie|undress|touch)\b|शरीर|फोटो|तस्वीर|चूम"
)


def stable_int(*parts: object) -> int:
    h = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def pick_weighted(rng: random.Random, items: list[tuple]) -> tuple:
    total = sum(i[3] for i in items)
    x = rng.random() * total
    acc = 0.0
    for it in items:
        acc += it[3]
        if x <= acc:
            return it
    return items[-1]


def build_spec(script: str, length: int, label: str, n: int, seed: int) -> dict:
    key = f"{script}:{length}:{label}:{n:06d}"
    rng = random.Random(stable_int(seed, key))
    subtype, desc, audience, _ = pick_weighted(rng, SUBTYPES[label])
    topic = rng.choice(TOPICS)
    target_gender = ""
    age = 0
    if label == "sexual_harassment":
        target_gender = "woman" if rng.random() < 0.75 else "man"
    elif label == "predatory_grooming":
        target_gender = "girl" if rng.random() < 0.5 else "boy"
    female_target = target_gender in ("woman", "girl")
    male_target = target_gender in ("man", "boy")
    pool_y = FEMALE if female_target else MALE if male_target else NAMES
    y = rng.choice(pool_y)
    x = rng.choice([n for n in (MALE if label in ("sexual_harassment", "predatory_grooming") else NAMES) if n != y])
    idx = 1 if script == "hindi_devanagari" else 0
    name_x, name_y = x[idx], y[idx]

    if label == "sexual_harassment":
        persona = f"{name_y} is an adult {target_gender} who is uncomfortable; {name_x} is the one making unwanted remarks."
        audience = f"adult_{target_gender}"
    elif label == "predatory_grooming":
        age = rng.randint(10, 15)
        persona = (
            f"{name_y} is a {age}-year-old {target_gender} in school (mention the class once). "
            f"{name_x} is an adult stranger, possibly claiming to be a teenager."
        )
        audience = f"minor_{target_gender}"
    elif label == "benign" and subtype in ("parent_child", "teacher_student"):
        persona = f"{name_y} is a school-age child; {name_x} is the parent or teacher."
        audience = "minor"
    elif label == "benign" and subtype == "consensual_flirt":
        persona = f"{name_x} and {name_y} are adults who like each other."
        audience = "adult"
    else:
        persona = f"{name_x} and {name_y} know each other."
        audience = {"peers": "peers", "adult": "adult", "minor": "minor"}.get(audience, audience)

    pair_line = ""
    pair_id = ""
    if label in PAIR_CONTEXT and n % 2 == 0:
        bank = PAIR_BANK[script]
        li = (n // 2) % len(bank)
        pair_line = bank[li]
        pair_id = f"{script}|{li}"
    return {
        "key": key,
        "n": n,
        "script": script,
        "length": length,
        "label": label,
        "label_id": CLASS_ID[label],
        "subtype": subtype,
        "subtype_desc": desc,
        "topic": topic,
        "audience": audience,
        "target_gender": target_gender,
        "child_age": age,
        "persona": persona,
        "name_x": name_x,
        "name_y": name_y,
        "pair_id": pair_id,
        "pair_line": pair_line,
    }


SYSTEM = (
    "You create synthetic chat conversations used to train safety classifiers that protect women "
    "and children from abuse. Follow the scenario exactly. Output JSON only, no commentary."
)


def build_prompt(spec: dict) -> str:
    k = spec["length"]
    x, y = spec["name_x"], spec["name_y"]
    first = x if k % 2 == 1 else y
    label = spec["label"]
    if spec["pair_line"]:
        last = (
            f'exactly this text and nothing else: "{spec["pair_line"]}". '
            + PAIR_CONTEXT[label].format(X=x, Y=y)
        )
    else:
        last = LAST_RULE[label].format(X=x, Y=y)
    rules = CLASS_RULES[label].format(X=x, Y=y)
    return (
        f"Write a chat of exactly {k} messages between {x} and {y}. Messages alternate; "
        f"message 1 is by {first}; message {k} (the last) is by {x}.\n"
        f"Language: {SCRIPT_INSTR[spec['script']]}\n"
        f"Setting: {spec['subtype_desc']} Topic: {spec['topic']}.\n"
        f"{spec['persona']}\n"
        f"Last message: {last}\n"
        f"Rules: {rules}\n"
        f"Each message under 30 words. Return only JSON: {{\"turns\": [{k} strings]}}"
    )


def split_for(spec: dict) -> str:
    if spec.get("pair_id"):
        script, idx = spec["pair_id"].split("|")
        m = int(idx) % 10
        return "test" if m == 0 else "val" if m == 1 else "train"
    h = stable_int("split", spec["key"]) % 100
    return "train" if h < 80 else "val" if h < 90 else "test"
