"""JSON payloads behind the HMM POS Tagger page."""
from nlp.hmm_pos import (BEST_HINGLISH_MIX, HINGLISH_MIXES, TAG_INFO, get_tagger, load_evaluation)
from nlp.langid import get_identifier
from nlp.preprocess import preprocess

MODEL_LABELS = {"en": "English", "hi": "Hindi", "hi-iiit": "Hindi (IIIT tagset)", "hg": "Hinglish"}


def model_key(lang, mix=None):
    if lang == "hg" and mix and mix in HINGLISH_MIXES and mix != BEST_HINGLISH_MIX:
        return f"hg:{mix}"
    return lang


def pos_tokens(text, lang):
    """Tokens the tagger sees: Phase 1 output (normalized, numbers as digits), punctuation kept."""
    steps = preprocess(text, "hi" if lang == "hi-iiit" else lang)
    return steps["final"]


def tag(text, lang=None, mix=None):
    if not lang:
        lang = get_identifier().predict(text)
    key = model_key(lang, mix)
    tagger = get_tagger(key)
    words = pos_tokens(text, lang)
    path, trellis = tagger.viterbi(words)
    columns = []
    for i, w in enumerate(words):
        sc = trellis["scores"][i]
        best = max(sc.values())
        columns.append({
            "word": w,
            "known": trellis["known"][i],
            "tag": path[i],
            # scores relative to the best cell in the column (0 = best)
            "scores": {t: sc[t] - best for t in tagger.tags},
            "back": trellis["back"][i],
        })
    return {
        "lang": lang, "model": key, "tags": tagger.tags, "words": words, "path": path,
        "columns": columns,
        "viterbi_logprob": trellis.get("best"),
        "forward_logprob": tagger.forward(words) if words else None,
        "tag_info": TAG_INFO,
    }


def model_info(lang, mix=None):
    tagger = get_tagger(model_key(lang, mix))
    tm = tagger.transition_matrix()
    return {
        "transitions": tm,
        "emissions": {t: [{"word": w, "p": p} for w, p in tagger.top_emissions(t, 8)] for t in tagger.tags},
        "tag_counts": dict(tagger.tag_count),
        "vocab": tagger.V,
        "train_tokens": sum(tagger.tag_count.values()),
        "tag_info": TAG_INFO,
    }


def evaluation():
    res = load_evaluation()
    res["mix_labels"] = HINGLISH_MIXES
    res["best_mix"] = BEST_HINGLISH_MIX
    return res
