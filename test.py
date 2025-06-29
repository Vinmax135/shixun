text = input()

import spacy

nlp = spacy.load("en_core_web_sm")

result = []
doc = nlp(text)
for token in doc:
        if token.dep_ in ("dobj", "pobj", "attr") and token.pos_ == "NOUN":
            modifiers = [child.text for child in token.children if child.dep_ == "amod"]
            phrase = " ".join(modifiers + [token.text])
            result.append(phrase.strip())

if len(result) > 0:
    result.append("item")