"""Published comparison targets; values are references, never observations."""

SOURCE = "https://papers.nips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf"
# time minutes, syntactic, semantic, total percent (Table 1, p.5).
TABLE1 = {
    "neg5-no-subsampling": (38, 63, 54, 59),
    "neg15-no-subsampling": (97, 63, 58, 61),
    "hs-no-subsampling": (41, 53, 40, 47),
    "nce5-no-subsampling": (38, 60, 45, 53),
    "neg5-subsampling": (14, 61, 58, 60),
    "neg15-subsampling": (36, 61, 61, 61),
    "hs-subsampling": (21, 52, 59, 55),
}
TABLE3 = dict(
    zip(
        (
            "neg5-no-subsampling",
            "neg5-subsampling",
            "neg15-no-subsampling",
            "neg15-subsampling",
            "hs-no-subsampling",
            "hs-subsampling",
        ),
        (24, 27, 27, 42, 19, 47),
        strict=True,
    )
)
SECTION4 = {"phrase-6b": 66, "phrase-33b": 72}
TABLE2 = [
    ("Newspapers", "New_York", "New_York_Times", "Baltimore", "Baltimore_Sun"),
    (
        "Newspapers",
        "San_Jose",
        "San_Jose_Mercury_News",
        "Cincinnati",
        "Cincinnati_Enquirer",
    ),
    ("NHL Teams", "Boston", "Boston_Bruins", "Montreal", "Montreal_Canadiens"),
    ("NHL Teams", "Phoenix", "Phoenix_Coyotes", "Nashville", "Nashville_Predators"),
    ("NBA Teams", "Detroit", "Detroit_Pistons", "Toronto", "Toronto_Raptors"),
    ("NBA Teams", "Oakland", "Golden_State_Warriors", "Memphis", "Memphis_Grizzlies"),
    ("Airlines", "Austria", "Austrian_Airlines", "Spain", "Spainair"),
    ("Airlines", "Belgium", "Brussels_Airlines", "Greece", "Aegean_Airlines"),
    ("Company executives", "Steve_Ballmer", "Microsoft", "Larry_Page", "Google"),
    ("Company executives", "Samuel_J._Palmisano", "IBM", "Werner_Vogels", "Amazon"),
]
TABLE4 = {
    "Vasco_de_Gama": ("Lingsugur", "Italian_explorer"),
    "Lake_Baikal": ("Great_Rift_Valley", "Aral_Sea"),
    "Alan_Bean": ("Rebbeca_Naomi", "moonwalker"),
    "Ionian_Sea": ("Ruegen", "Ionian_Islands"),
    "chess_master": ("chess_grandmaster", "Garry_Kasparov"),
}
TABLE5 = {
    "Czech currency": ("koruna", "Check_crown", "Polish_zolty", "CTK"),
    "Vietnam capital": ("Hanoi", "Ho_Chi_Minh_City", "Viet_Nam", "Vietnamese"),
    "German airlines": (
        "airline_Lufthansa",
        "carrier_Lufthansa",
        "flag_carrier_Lufthansa",
        "Lufthansa",
    ),
    "Russian river": ("Moscow", "Volga_River", "upriver", "Russia"),
    "French actress": (
        "Juliette_Binoche",
        "Vanessa_Paradis",
        "Charlotte_Gainsbourg",
        "Cecile_De",
    ),
}
# Columns Redmond, Havel, ninjutsu, graffiti, capitulate; empty = published OOV.
TABLE6 = {
    "Collobert (50d; 2 months)": [
        ["conyers", "lubbock", "keene"],
        ["plauen", "dzerzhinsky", "osterreich"],
        ["reiki", "kohona", "karate"],
        ["cheesecake", "gossip", "dioramas"],
        ["abdicate", "accede", "rearm"],
    ],
    "Turian (200d; few weeks)": [
        ["McCarthy", "Alston", "Cousins"],
        ["Jewell", "Arzu", "Ovitz"],
        [],
        ["gunfire", "emotion", "impunity"],
        [],
    ],
    "Mnih (100d; 7 days)": [
        ["Podhurst", "Harlang", "Agarwal"],
        ["Pontiff", "Pinochet", "Rodionov"],
        [],
        ["anaesthetics", "monkeys", "Jews"],
        ["Mavericks", "planning", "hesitated"],
    ],
    "Skip-Phrase (1000d; 1 day)": [
        ["Redmond_Wash.", "Redmond_Washington", "Microsoft"],
        ["Vaclav_Havel", "president_Vaclav_Havel", "Velvet_Revolution"],
        ["ninja", "martial_arts", "swordsmanship"],
        ["spray_paint", "grafitti", "taggers"],
        ["capitulation", "capitulated", "capitulating"],
    ],
}
TABLE6_QUERIES = ("Redmond", "Havel", "ninjutsu", "graffiti", "capitulate")
