
import argparse, os, re, sys, socket, time, string
from typing import List, Tuple, Set, Optional, Dict

# ==============================
# 0) CSV LOADERS
# ==============================

def load_gloss_set(csv_path: str) -> Set[str]:
    """Load a set of canonical gloss labels from a CSV (UPPERCASE)."""
    import csv
    s: Set[str] = set()
    if not csv_path or not os.path.exists(csv_path): return s
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames: return s
        headers = {h.lower().strip(): h for h in r.fieldnames}
        cands = [
            "annotation id gloss","annotation_id_gloss","idgloss","id_gloss",
            "lemma id gloss","lemma_id_gloss","signbank id gloss","signbank_id_gloss",
            "gloss","class label","class_label","main entry","main_entry",
            "main entry label","main_entry_label","label","name"
        ]
        col = next((headers[c] for c in cands if c in headers), None)
        for row in r:
            g = (row.get(col) or "").strip().upper() if col else ""
            if g: s.add(g)
    return s

def load_gloss_map(csv_path: str) -> Dict[str, str]:
    """Load English → ID-GLOSS mapping (LOWERCASE keys, UPPERCASE values)."""
    import csv
    m: Dict[str, str] = {}
    if not csv_path or not os.path.exists(csv_path): return m
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames: return m
        headers = {h.lower().strip(): h for h in r.fieldnames}
        en_col = next((headers[c] for c in ["english","en","phrase","token"] if c in headers), None)
        gl_col = next((headers[c] for c in ["gloss","idgloss","label","id_gloss"] if c in headers), None)
        for row in r:
            en = (row.get(en_col) or "").strip().lower() if en_col else ""
            gl = (row.get(gl_col) or "").strip().upper() if gl_col else ""
            if en and gl: m[en] = gl
    return m

def load_sigml_map(csv_path: str) -> Dict[str, str]:
    """
    Load GLOSS -> SiGML content.
    Columns (auto-detected): gloss, sigml_file, sigml_inline
    If both present, inline takes precedence.
    """
    import csv
    m: Dict[str, str] = {}
    if not csv_path or not os.path.exists(csv_path): return m
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        headers = {h.lower().strip(): h for h in (r.fieldnames or [])}
        gloss_col = next((headers[c] for c in ["gloss","idgloss","id_gloss","label","name"] if c in headers), None)
        file_col  = next((headers[c] for c in ["sigml_file","file","path"] if c in headers), None)
        text_col  = next((headers[c] for c in ["sigml_inline","sigml","xml"] if c in headers), None)
        for row in r:
            g = (row.get(gloss_col) or "").strip().upper() if gloss_col else ""
            if not g: continue
            text = (row.get(text_col) or "").strip() if text_col else ""
            if not text and file_col and row.get(file_col):
                fp = row[file_col].strip()
                if fp and os.path.exists(fp):
                    with open(fp, "r", encoding="utf-8") as fh:
                        text = fh.read()
            if text:
                m[g] = text
    return m

# ==============================
# 1) TRANSCRIPTION (faster-whisper)
# ==============================

def transcribe_with_whisper(audio_path: str, model_size: str = "small") -> str:
    """
    Transcribe English speech using faster-whisper (CTranslate2).
    Chooses device/compute automatically:
      - GPU available → device="cuda", compute_type="float16"
      - otherwise     → device="cpu",  compute_type="int8"
    """
    from faster_whisper import WhisperModel
    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False

    device = "cuda" if use_gpu else "cpu"
    compute_type = "float16" if use_gpu else "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(audio_path, language="en")
    return " ".join(seg.text for seg in segments).strip()

# ==============================
# 2) ENGLISH → ASL GLOSS
# ==============================

CONTRACTIONS = {
    "i'm":"i am","you're":"you are","he's":"he is","she's":"she is","it's":"it is","we're":"we are","they're":"they are",
    "i've":"i have","you've":"you have","we've":"we have","they've":"they have",
    "i'd":"i would","you'd":"you would","he'd":"he would","she'd":"she would","we'd":"we would","they'd":"they would",
    "can't":"can not","won't":"will not","isn't":"is not","aren't":"are not","wasn't":"was not","weren't":"were not",
    "don't":"do not","doesn't":"does not","didn't":"did not","shouldn't":"should not","wouldn't":"would not","couldn't":"could not",
    "there's":"there is","that's":"that is","what's":"what is","who's":"who is",
}
PRONOUN_MAP = {"i":"ME","me":"ME","my":"MY","mine":"MY","you":"YOU","your":"YOUR","yours":"YOUR","he":"HE","him":"HE","his":"HIS",
               "she":"SHE","her":"HER","we":"WE","us":"WE","our":"OUR","ours":"OUR","they":"THEY","them":"THEY","their":"THEIR","theirs":"THEIR","it":"IT"}
LEXICON = {"go":"GO","going":"GO","went":"GO","gone":"GO","want":"WANT","like":"LIKE","need":"NEED","have":"HAVE","get":"GET","give":"GIVE",
           "take":"TAKE","eat":"EAT","drink":"DRINK","see":"SEE","look":"LOOK","work":"WORK","study":"STUDY","help":"HELP","bring":"BRING",
           "make":"MAKE","buy":"BUY","play":"PLAY","think":"THINK","know":"KNOW","learn":"LEARN","live":"LIVE","come":"COME","leave":"LEAVE",
           "move":"MOVE","happy":"HAPPY","sad":"SAD","tired":"TIRED","angry":"ANGRY","good":"GOOD","bad":"BAD"}
ARTICLES = {"a","an","the"}; BE_FORMS = {"am","is","are","was","were","be","been","being"}
DO_SUPPORT = {"do","does","did"}; MODALS_FUTURE = {"will"}; NEGATORS = {"not","no","never"}
TIME_REGEX = re.compile(r"\b(yesterday|today|tomorrow|now|tonight|this (morning|afternoon|evening|week|month|year)|last (night|week|month|year)|next (week|month|year)|on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)|in the (morning|afternoon|evening))\b", re.IGNORECASE)
WH_WORDS = {"who","what","where","when","why","how"}

def expand_contractions(t: str) -> str:
    t = t.lower()
    for k,v in CONTRACTIONS.items(): t = re.sub(rf"\b{k}\b", v, t)
    return t

def split_sentences(t: str) -> List[str]:
    parts = re.split(r"([.?!])", t); out=[]
    for i in range(0,len(parts),2):
        s = parts[i].strip(); p = parts[i+1] if i+1<len(parts) else ""
        if s: out.append(s+p)
    return out or ([t] if t else [])

def extract_time_phrase(s: str) -> Tuple[str,str]:
    m = TIME_REGEX.search(s)
    if not m: return ("", s)
    a,b = m.span(); return (s[a:b].upper(), (s[:a]+s[b:]).strip())

def _match_phrase(tokens: List[str], start: int, en2gloss: Dict[str,str], max_n=4):
    for n in range(min(max_n, len(tokens)-start), 1, -1):
        ph = " ".join(tokens[start:start+n]).lower()
        if ph in en2gloss: return en2gloss[ph], n
    return (None, 0)

def english_to_asl_gloss(text: str, library: Optional[Set[str]]=None, en2gloss: Optional[Dict[str,str]]=None) -> str:
    library = library or set(); en2gloss = en2gloss or {}
    sents = split_sentences(expand_contractions(text)); out=[]
    for s in sents:
        is_q = s.endswith("?"); s = s[:-1] if is_q else s
        time_g, core = extract_time_phrase(s)
        core = re.sub(r"[,:;\"()\[\]]"," ",core); core = re.sub(r"\s+"," ",core).strip()
        toks = [t.lower() for t in core.split() if t]
        neg=False; fut=False; proc=[]; i=0
        while i<len(toks):
            t=toks[i]; T=t.upper()
            if t in ARTICLES: i+=1; continue
            if t=="going" and i+1<len(toks) and toks[i+1]=="to": fut=True; i+=2; continue
            if t in MODALS_FUTURE: fut=True; i+=1; continue
            if t in BE_FORMS or t in DO_SUPPORT: i+=1; continue
            if t in NEGATORS: neg=True; i+=1; continue
            gN, n = _match_phrase(toks,i,en2gloss,4)
            if gN: proc.append(gN); i+=n; continue
            if t in en2gloss: proc.append(en2gloss[t]); i+=1; continue
            if T in library: proc.append(T); i+=1; continue
            if t in PRONOUN_MAP: proc.append(PRONOUN_MAP[t]); i+=1; continue
            if t in LEXICON: proc.append(LEXICON[t]); i+=1; continue
            if re.match(r".+ed$", t) and len(t)>3:
                base = re.sub(r"ed$","",t); base = LEXICON.get(base, base.upper())
                proc.extend(["FINISH", base]); i+=1; continue
            if t=="of": i+=1; continue
            proc.append(T); i+=1
        parts=[]
        if time_g: parts.append(time_g)
        if fut: parts.append("FUTURE")
        parts.extend(proc)
        if is_q:
            wh_idx=[k for k,tk in enumerate(parts) if tk.lower() in WH_WORDS]
            if wh_idx:
                tok=parts.pop(wh_idx[0]); parts.append(tok); parts.append("Q")
            else: parts.append("Q")
        if neg: parts.append("NOT")
        out.append(" ".join([p for p in parts if p]))
    return " / ".join(out)

# ==============================
# 3) AVATAR SOCKET (SiGML Player on TCP 8052)
# ==============================

def send_sigml_to_player(sigml_text: str, host: str="127.0.0.1", port: int=8052, encoding: str="utf-8"):
    """Send SiGML text to the CWASA/JASigning SiGML Player over TCP."""
    data = sigml_text.encode(encoding)
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(data)

def drive_avatar_sequence(gloss_tokens: List[str],
                          sigml_lookup: Dict[str,str],
                          host="127.0.0.1", port=8052,
                          per_sign_wait: float=1.6,
                          skip_unknown=True) -> List[str]:
    """Send one SiGML snippet per gloss; wait a fixed time between sends."""
    sent=[]; missing=[]
    for g in gloss_tokens:
        sig = sigml_lookup.get(g.upper())
        if not sig:
            missing.append(g)
            if skip_unknown: continue
        if sig:
            send_sigml_to_player(sig, host=host, port=port)
            sent.append(g)
            time.sleep(per_sign_wait)
    if missing:
        print("[warn] No SiGML for:", ", ".join(sorted(set(missing))))
    return sent

def pick_gloss_tokens_for_avatar(gloss_str: str) -> List[str]:
    """Split gloss into tokens suitable for avatar driving."""
    toks=[]
    for chunk in gloss_str.replace("/", " ").split():
        w = re.sub(r"[^A-Z\-]", "", chunk.upper())
        if not w: continue
        if w in {"Q"}:  # skip yes/no marker unless you model it
            continue
        toks.append(w)
    return toks

# Fingerspelling support
def load_alphabet_sigml(dir_path: str) -> dict[str, str]:
    """Load A..Z .sigml files for fingerspelling."""
    lookup = {}
    if not dir_path or not os.path.isdir(dir_path):
        return lookup
    for L in string.ascii_uppercase:
        fp = os.path.join(dir_path, f"{L}.sigml")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as fh:
                lookup[L] = fh.read()
    return lookup

def fingerspell(word: str, alpha_sigml: dict[str, str],
                host="127.0.0.1", port=8052, per_letter_wait: float = 0.6):
    """Send per-letter SiGML for a word (A..Z only)."""
    for ch in word.upper():
        sig = alpha_sigml.get(ch)
        if not sig:
            continue
        send_sigml_to_player(sig, host=host, port=port)
        time.sleep(per_letter_wait)

# ==============================
# 4) ORCHESTRATION / CLI
# ==============================

def run(audio_path: str,
        model_size: str="small",
        gloss_csv: str="",
        gloss_map_csv: str="",
        send_to_avatar: bool=False,
        sigml_map_csv: str="",
        avatar_host: str="127.0.0.1",
        avatar_port: int=8052,
        per_sign_wait: float=1.6,
        alphabet_sigml_dir: str="",
        fingerspell_unknown: bool=False) -> None:

    library = load_gloss_set(gloss_csv) if gloss_csv else set()
    en2gloss = load_gloss_map(gloss_map_csv) if gloss_map_csv else {}
    transcript = transcribe_with_whisper(audio_path, model_size=model_size)
    gloss = english_to_asl_gloss(transcript, library=library, en2gloss=en2gloss)

    flags=[]
    if library: flags.append("library")
    if en2gloss: flags.append("map")
    print("\n=== Transcript ===\n"+transcript)
    print(f"\n=== ASL GLOSS{' ('+' + '.join(flags)+')' if flags else ''} ===\n"+gloss)

    if send_to_avatar:
        if not sigml_map_csv:
            print("\n[error] --send_to_avatar needs --sigml_map_csv pointing to your GLOSS→SiGML mapping.")
            return
        sigml_lookup = load_sigml_map(sigml_map_csv)
        if not sigml_lookup:
            print(f"\n[error] No SiGML entries loaded from: {sigml_map_csv}")
            return

        tokens = pick_gloss_tokens_for_avatar(gloss)
        known = [g for g in tokens if g.upper() in sigml_lookup]
        missing = [g for g in tokens if g.upper() not in sigml_lookup]

        # send known signs
        if known:
            print("\n[avatar] known tokens →", known)
            drive_avatar_sequence(known, sigml_lookup,
                                  host=avatar_host, port=avatar_port, per_sign_wait=per_sign_wait)

        # fingerspell unknowns
        if fingerspell_unknown and missing and alphabet_sigml_dir:
            alpha = load_alphabet_sigml(alphabet_sigml_dir)
            if alpha:
                print("[avatar] fingerspelling →", missing)
                for g in missing:
                    if g in {"Q"}:  # skip function markers
                        continue
                    fingerspell(g, alpha, host=avatar_host, port=avatar_port, per_letter_wait=0.6)
            else:
                print(f"[warn] No alphabet .sigml found in {alphabet_sigml_dir}")

def main():
    ap = argparse.ArgumentParser(description="Speech → English → ASL GLOSS → (optional) 3D avatar via SiGML")
    ap.add_argument("audio", help="Path to audio file (wav, mp3, m4a, etc.)")
    ap.add_argument("--model", default="small", help="faster-whisper model: tiny|base|small|medium|large-v3, etc.")
    ap.add_argument("--gloss_csv", default="", help="CSV of ID gloss labels (ASL-LEX / WLASL / Signbank export)")
    ap.add_argument("--gloss_map_csv", default="", help="CSV (english,gloss) for explicit EN→IDG mapping (phrases allowed)")
    ap.add_argument("--send_to_avatar", action="store_true", help="Send SiGML to CWASA/JASigning SiGML Player (port 8052)")
    ap.add_argument("--sigml_map_csv", default="", help="CSV mapping GLOSS→SiGML (columns: gloss, sigml_file OR sigml_inline)")
    ap.add_argument("--avatar_host", default="127.0.0.1", help="SiGML Player host")
    ap.add_argument("--avatar_port", type=int, default=8052, help="SiGML Player TCP port")
    ap.add_argument("--per_sign_wait", type=float, default=1.6, help="Seconds to wait between signs when streaming sequentially")
    ap.add_argument("--alphabet_sigml_dir", default="", help="Folder with A..Z .sigml files for fingerspelling")
    ap.add_argument("--fingerspell_unknown", action="store_true", help="Fingerspell gloss tokens missing from --sigml_map_csv")
    args = ap.parse_args()

    try:
        if not os.path.exists(args.audio):
            raise FileNotFoundError(f"Audio not found: {args.audio}")
        run(args.audio, model_size=args.model,
            gloss_csv=args.gloss_csv, gloss_map_csv=args.gloss_map_csv,
            send_to_avatar=args.send_to_avatar, sigml_map_csv=args.sigml_map_csv,
            avatar_host=args.avatar_host, avatar_port=args.avatar_port, per_sign_wait=args.per_sign_wait,
            alphabet_sigml_dir=args.alphabet_sigml_dir, fingerspell_unknown=args.fingerspell_unknown)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
