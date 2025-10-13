from pathlib import Path 
version = "V1"
default_dpi = 400
# Path to the project root directory

projectDir = Path(version)
projectDir.mkdir(exist_ok=True, parents=True)

dataDir = projectDir / "data"
outputDir = projectDir / "output"
PaperDir = projectDir / "paper"

dataDir.mkdir(exist_ok=True, parents=True)
outputDir.mkdir(exist_ok=True, parents=True)
PaperDir.mkdir(exist_ok=True, parents=True)


qts = [
    "AG",
    "ALP",
    "ALT",
    "APTT",
    "AST",
    "Alb",
    "BMI",
    "BNP",
    "C1q",
    "Cl",
    "CK",
    "CK-MB",
    "Cr",
    "D-Bil",
    "D-Dimer",
    "eGFR(CKD-EPI)",
    "FBG",
    "FFA",
    "FT3",
    "FT4",
    "GGT",
    "Glu",
    "HCT",
    "HDLC",
    "Hb",
    "HbA1C%",
    "Hcy",
    "Height",
    "INR",
    "Isn",
    "K",
    "LDH",
    "LDLC",
    "LPa",
    "LYM",
    "LYM%",
    "MCH",
    "MCHC",
    "MCV",
    "MPV",
    "NE",
    "NE%",
    "Na",
    "PCT",
    "PLT",
    "RBC",
    "T-Bil",
    "T3",
    "T4",
    "TC",
    "TG",
    "TP",
    "TSH",
    "UA",
    "Urea",
    "WBC",
    "Weight",
    "hs-CRP",
    "hsTnI",
    "nonHDL",
    "sdLDL",
] + ["Age"]
bts = ["Sex"]
System_traits_map = {
    "Anthropometrics": ["BMI", "Height", "Weight", "Age"],
    "Cardiovascular": [
        "BNP",
        "CK",
        "CK-MB",
        "D-Dimer",
        "EF",
        "HDLC",
        "Hcy",
        "LDH",
        "LDLC",
        "LPa",
        "TC",
        "TG",
        "hs-CRP",
        "hsTnI",
        "nonHDL",
        "sdLDL",
    ],
    "Digestive": [
        "AG",
        "ALP",
        "ALT",
        "AST",
        "Alb",
        "D-Bil",
        "GGT",
        "LDH",
        "T-Bil",
        "TP",
    ],
    "Electrolytes": ["Cl", "K", "Na"],
    "Endocrine": [
        "FBG",
        "FFA",
        "FT3",
        "FT4",
        "Glu",
        "HbA1C%",
        "Isn",
        "T3",
        "T4",
        "TSH",
    ],
    "Hematology": [
        "APTT",
        "D-Dimer",
        "HCT",
        "Hb",
        "INR",
        "MCH",
        "MCHC",
        "MCV",
        "MPV",
        # "PDW",
        "PLT",
        "RBC",
    ],
    "Immune": [
        "C1q",
        "LYM",
        "LYM%",
        "NE",
        "NE%",
        "PCT",
        "WBC",
        "hs-CRP",
    ],
    "Renal": ["Cr", "UA", "Urea", "eGFR(CKD-EPI)"],
}

TableDir = PaperDir / "Table"
FigureDir = PaperDir / "Figure"
RawDataDir = PaperDir / "RawData"
TableDir.mkdir(exist_ok=True, parents=True)
FigureDir.mkdir(exist_ok=True, parents=True)
RawDataDir.mkdir(exist_ok=True, parents=True)