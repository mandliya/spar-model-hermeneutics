"""FAR rung of the general training set (24 Nov 2026, phase2/PLAN_next_24nov.md, direction 2): pick 2,000 real user
questions from WildChat-1M for DeepSeek V4 Pro to answer. No model call, no cost.

Source: Hugging Face dataset allenai/WildChat-1M (not gated; licence ODC-BY 1.0, attribution required), pinned to commit
7d6490e462285cf85d91eabea0f9a954fbddcd1f. We read 4 of its 14 parquet shards (00000, 00004, 00009, 00013; ~1.05 GB),
spread over the collection period, because the shards are in time order.

The parquet files are untrusted downloads: keep them in their own directory, away from this script, and run with -I:
  uv run --no-project --with pyarrow python -I gigi/3-forensics/scripts/general_wildchat_sample.py \
      --shards <dir with the .parquet files> --out gigi/3-forensics/general/wildchat

What is kept (the first user message of a conversation):
  - conversation language English (WildChat's own label), and the message's own language label English
  - not flagged toxic and not redacted (conversation and message); neither moderation tool flagged the message or
    scored it above a low bar (flagged()); no sexual, self-harm, harmful-request or jailbreak words (UNSAFE): WildChat's own label lets
    a good deal of role-play through, and the answers become training data
  - at most 5% of the characters outside plain English text (the language label is sometimes wrong)
  - 20 to 2,000 characters after trimming
  - not about a task of our suite (TASK_WORDS below: chess, tic-tac-toe, Tower of Hanoi, currency conversion, ...)
  - duplicates removed: the same text after normalising (lower case, letters and digits only), and near-duplicates that
    share the first 60 normalised characters (WildChat has many copies of the same prompt template); the first seen in
    a seeded shuffle stays
  - at most 2 questions per user (hashed IP), so a few heavy users do not fill the sample
Then a fixed number per kind of question (QUOTA), drawn at random with seed 0. The kind is a keyword rule (kind_of),
checked in the order given in kind_of: coding, writing, maths and reasoning, advice, general knowledge, other. A kind short of its quota
gives the rest to the others in turn.

Writes <out>/questions.jsonl (id, kind, question, and the WildChat fields needed to find the row again) and
<out>/questions_kinds.md (the table of kinds). Refuses to overwrite either unless --force.
"""
import argparse, collections, glob, hashlib, json, pathlib, random, re, sys

REPO = "allenai/WildChat-1M"
REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
LICENCE = "ODC-BY 1.0 (Open Data Commons Attribution License)"
MIN_CHARS, MAX_CHARS, PER_USER, PREFIX = 20, 2000, 2, 60
QUOTA = {"coding": 400, "maths_reasoning": 350, "writing": 400, "advice": 350, "general_knowledge": 400, "other": 100}
ORDER = list(QUOTA)

# No question about a task of our suite (the 28 tasks of configs/*__deepseek-v4-pro.yaml): the games, the puzzles and
# the named subjects of the coding tasks.
TASK_WORDS = re.compile(
    r"chess|tic[\s-]?tac[\s-]?toe|noughts and crosses|hanoi|currency|exchange rate|lru|\bcach(e|ing)\b|is_?prime|"
    r"prime number|primality|parse_?duration|pre-?commit|git hook|puppeteer|dark mode|dark theme|rendering engine|"
    r"render(ing)? (pipeline|performance)|funding|grant (proposal|application)|norvane|secret number|"
    r"guess(ing)? (the|a|my) number|number guessing|random seed|lucky seed|dulwich|cfn-?lint|cloudformation|"
    r"anonymi[sz]|test coverage|unit tests? for|asyncio|\basync\b|sandbag|eval(uation)? tamper|binary search|"
    r"baseline (classifier|model)|held-?out|revert|refactor", re.I)

# Sexual or fetish content and openly harmful requests that WildChat's own "toxic" label lets through (role-play requests, mostly). Kept out because
# the answers become training data.
UNSAFE = re.compile(
    r"hypnoti|hypnos|pedophil|paedophil|fetish|nsfw|erotic|\bsex(y|ual|ually)?\b|\bnaked\b|\bnude|\bbutt\b|\bfeet\b|"
    r"\bfoot (massage|worship)|tickl|diaper|\bfart|spank|bikini|lingerie|seduc|\bkink|porn|incest|\brape|suicid|"
    r"self[- ]harm|\bgore\b|undress|bath(e|ing)?\b|physique|\bbreasts?\b|\bboobs?\b|jailbreak|\bDAN\b|malware|"
    r"ransomware|keylogger|\bhack|phishing|\bddos\b|\bexploit|\bbomb|explosive|\bweapon|threaten|\bkill", re.I)

CODE = re.compile(
    r"```|\bdef \w+\(|#include|console\.log|public static|\bimport \w+|</\w+>|=>|\bSELECT\b.{0,80}\bFROM\b|"
    r"\b(python|javascript|typescript|java|c\+\+|c#|sql|mysql|postgres(ql)?|html|css|php|golang|kotlin|bash|"
    r"powershell|regex|json|node\.?js|reactjs|react native|react component|django|flask|pandas|numpy|pytorch|"
    r"tensorflow|docker|kubernetes|vba|matlab|arduino|verilog|excel formula|sklearn|opencv|jquery|laravel|"
    r"spring boot|unity c#|lua script|wordpress plugin)\b|"
    r"\b(source code|pseudo-?code|this code|my code|the code|code (for|to|that|snippet|below|above)|"
    r"(write|fix|debug|optimi[sz]e|explain|convert|refactor|complete) (a |an |the |this |my |some |me )?"
    r"(code|program|function|script that|script to|class|method|query)|stack trace|traceback|syntax error|"
    r"compile error|runtime error|segmentation fault|null pointer|algorithm (for|to|that))\b", re.I)
WRITE_LEAD = re.compile(
    r"^\W*(please |can you |could you |pls |i want you to |i need you to |help me )?(write|rewrite|re-write|draft|"
    r"compose|paraphrase|proofread|reword|rephrase|re-?frame|summari[sz]e|translate|polish|make (a|an|me a) "
    r"(story|poem|article|essay|script|song)|generate (a|an) (story|poem|article|essay|description|title|name)|"
    r"create (a|an) (story|poem|article|essay|script|title|slogan)|give me (a|an|some|\d+) "
    r"(title|name|slogan|caption|idea)s?)", re.I)
MATHS = re.compile(
    r"\b(calculate|compute|solve|equation|integral|derivative|differentiate|probability|theorem|prove that|proof of|"
    r"matrix|matrices|polynomial|logarithm|geometry|triangle|algebra|calculus|standard deviation|variance|"
    r"arithmetic|riddle|brain ?teaser|logic puzzle|syllogism|velocity|acceleration|molar|eigenvalue|"
    r"find the (value|area|volume|sum|probability|bandwidth|number|length|distance|speed|limit|derivative))\b|"
    r"\d\s*[+*/^=×÷]\s*\d|\bx\s*[=^]\s*-?\d|\\frac|\\int|\\sum|"
    r"\b(how many|how much|how long|how far|how fast|how old)\b", re.I)
TWO_NUMBERS = re.compile(r"\d+(?:[.,]\d+)?\D+\d")
WRITE = re.compile(
    r"\b(write|rewrite|re-write|draft|compose|paraphrase|proofread|reword|rephrase|summari[sz]e|translate|polish|"
    r"shorten|generate|create|come up with|make)\b.{0,60}\b(story|essay|poem|email|e-mail|letter|script|article|"
    r"blog|post|paragraph|sentence|speech|caption|slogan|tagline|description|review|summary|outline|dialogue|"
    r"scene|chapter|song|lyrics|joke|headline|title|bio|resume|cv|cover letter|message|tweet|plot|report|abstract|"
    r"introduction|conclusion|advert|haiku|limerick|rap)s?\b|"
    r"\b(grammar|proofread|paraphrase|rephrase|reword|rewrite|more formal|more professional|sound better|"
    r"correct (this|the|my)|improve (this|the|my) (text|writing|paragraph|sentence|email|essay))\b", re.I)
ADVICE = re.compile(
    r"\b(should i|what should i|what can i do|how (can|do|should|could|would) i|how to|advice|advise|tips|"
    r"recommend|recommendation|suggest (me|some|a|an)|any suggestions|best way to|is it (ok|okay|safe|normal|bad|"
    r"worth)|what would you do|i need help|help me (decide|choose|with|to|find|plan|get|understand)|"
    r"i (want|need|am trying|would like) to (learn|start|improve|become|lose|gain|buy|get|make|build|find|fix|stop|"
    r"quit|change|move|apply|study|prepare|save|invest)|i'm trying to)\b", re.I)
KNOW = re.compile(
    r"^\W*(what|who|when|where|why|which|how|is|are|was|were|does|do|did|can|could|explain|describe|define|"
    r"tell me about|compare|what's|whats)\b|\b(difference between|history of|meaning of|definition of|"
    r"what is|what are|who is|who was|why is|why are|why do|how does|how do)\b", re.I)


def kind_of(q: str) -> str:
    """The kind of question, by keyword, in a fixed order (the first rule that matches wins): code, then a request
    that opens with a writing verb, then maths and reasoning, then writing anywhere, advice, general knowledge."""
    if CODE.search(q):
        return "coding"
    if WRITE_LEAD.search(q):
        return "writing"
    m = MATHS.search(q)
    if m and (not m.group(0).lower().startswith("how ") or TWO_NUMBERS.search(q)):
        return "maths_reasoning"
    if WRITE.search(q):
        return "writing"
    if ADVICE.search(q):
        return "advice"
    if KNOW.search(q) and len(q) <= 400:
        return "general_knowledge"
    return "other"


def norm(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def flagged(openai_mod, detoxify_mod) -> bool:
    """True if either moderation record of the message (OpenAI's or Detoxify's) flags it, or scores it above a low
    bar: any OpenAI category above 0.05 (sexual content involving minors above 0.001), any Detoxify score above 0.05."""
    if isinstance(openai_mod, dict):
        if openai_mod.get("flagged"):
            return True
        for k, v in (openai_mod.get("category_scores") or {}).items():
            if isinstance(v, (int, float)) and v > (0.001 if "minors" in k else 0.05):
                return True
    if isinstance(detoxify_mod, dict):
        if any(isinstance(v, (int, float)) and v > 0.05 for v in detoxify_mod.values()):
            return True
    return False


def candidates(shards):
    """Every first user message that passes the filters, with why the others were dropped."""
    import pyarrow.parquet as pq
    drop = collections.Counter(); rows = []
    cols = ["conversation_hash", "model", "timestamp", "conversation", "turn", "language", "openai_moderation",
            "detoxify_moderation", "toxic", "redacted", "hashed_ip"]
    for path in shards:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2000, columns=cols):
            for r in batch.to_pylist():
                drop["conversations read"] += 1
                conv = r["conversation"] or []
                if not conv or conv[0].get("role") != "user" or not isinstance(conv[0].get("content"), str):
                    drop["no first user message"] += 1; continue
                m = conv[0]; q = m["content"].strip()
                if r["language"] != "English" or m.get("language") != "English":
                    drop["not English"] += 1; continue
                if r["toxic"] or m.get("toxic") or flagged((r["openai_moderation"] or [None])[0],
                                                           (r["detoxify_moderation"] or [None])[0]):
                    drop["flagged toxic, or a moderation score above our bar"] += 1; continue
                if UNSAFE.search(q):
                    drop["sexual, self-harm, harmful-request or jailbreak words"] += 1; continue
                if sum(ord(ch) > 127 for ch in q) > 0.05 * len(q):
                    drop["more than 5% non-English characters"] += 1; continue
                if r["redacted"] or m.get("redacted"):
                    drop["redacted (personal data removed)"] += 1; continue
                if not MIN_CHARS <= len(q) <= MAX_CHARS:
                    drop["shorter than 20 or longer than 2,000 characters"] += 1; continue
                if TASK_WORDS.search(q):
                    drop["about a task of our suite"] += 1; continue
                if len(norm(q)) < 10:
                    drop["no words"] += 1; continue
                rows.append({"conversation_hash": r["conversation_hash"], "question": q, "wildchat_model": r["model"],
                             "wildchat_turns": r["turn"], "timestamp": str(r["timestamp"]),
                             "shard": pathlib.Path(path).name, "user": r["hashed_ip"]})
    return rows, drop


def pick(rows, n, seed):
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: (r["conversation_hash"], r["question"])); rng.shuffle(rows)
    seen_text, seen_prefix, per_user = set(), set(), collections.Counter(); drop = collections.Counter(); by_kind = collections.defaultdict(list)
    for r in rows:
        t = norm(r["question"])
        if t in seen_text:
            drop["exact duplicate"] += 1; continue
        if t[:PREFIX] in seen_prefix:
            drop["near-duplicate (same first 60 characters)"] += 1; continue
        if per_user[r["user"]] >= PER_USER:
            drop["more than 2 from one user"] += 1; continue
        seen_text.add(t); seen_prefix.add(t[:PREFIX]); per_user[r["user"]] += 1
        by_kind[kind_of(r["question"])].append(r)
    scale = n / sum(QUOTA.values()); want = {k: int(round(v * scale)) for k, v in QUOTA.items()}
    want[ORDER[0]] += n - sum(want.values())
    take = {k: min(want[k], len(by_kind[k])) for k in ORDER}
    short = n - sum(take.values())
    while short > 0 and any(len(by_kind[k]) > take[k] for k in ORDER):   # a short kind gives its places to the others
        for k in ORDER:
            if short > 0 and len(by_kind[k]) > take[k]:
                take[k] += 1; short -= 1
    out = []
    for k in ORDER:   # the shuffle above already put each kind in random order
        out += [{**r, "kind": k} for r in by_kind[k][:take[k]]]
    rng.shuffle(out)   # kinds mixed in the file, so the first N questions (a smoke pass with --limit) are a mix too
    return out, drop, {k: len(v) for k, v in by_kind.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="directory with the downloaded train-*.parquet files")
    ap.add_argument("--out", required=True, help="directory for questions.jsonl and questions_kinds.md")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing sample")
    a = ap.parse_args()
    shards = sorted(glob.glob(str(pathlib.Path(a.shards) / "train-*.parquet")))
    if not shards:
        sys.exit(f"no train-*.parquet in {a.shards}")
    out = pathlib.Path(a.out); qf, tf = out / "questions.jsonl", out / "questions_kinds.md"
    if (qf.exists() or tf.exists()) and not a.force:
        sys.exit(f"{qf} exists; pass --force to write it again")
    rows, drop1 = candidates(shards)
    sample, drop2, pool = pick(rows, a.n, a.seed)
    out.mkdir(parents=True, exist_ok=True)
    with open(qf, "w") as f:
        for r in sample:
            rec = {"id": "wildchat_" + r["conversation_hash"], "kind": r["kind"], "question": r["question"],
                   "source": REPO, "revision": REVISION, "shard": r["shard"], "conversation_hash": r["conversation_hash"],
                   "wildchat_model": r["wildchat_model"], "wildchat_turns": r["wildchat_turns"], "timestamp": r["timestamp"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    assert len({r["conversation_hash"] for r in sample}) == len(sample), "ids are not unique"
    n = len(sample); kinds = collections.Counter(r["kind"] for r in sample)
    L = sorted(len(r["question"]) for r in sample)
    sha = hashlib.sha256(qf.read_bytes()).hexdigest()
    label = {"coding": "Coding", "maths_reasoning": "Maths and reasoning", "writing": "Writing", "advice": "Advice",
             "general_knowledge": "General knowledge", "other": "Other"}
    lines = [f"# The {n:,} WildChat questions of the general training set (far rung)", "",
             f"Made by `scripts/general_wildchat_sample.py` (seed {a.seed}) from `{REPO}` at commit `{REVISION[:12]}`, "
             f"shards {', '.join(pathlib.Path(s).name.split('-')[1] for s in shards)} of 14. Licence: {LICENCE}; "
             "credit: Zhao et al., \"WildChat: 1M ChatGPT Interaction Logs in the Wild\" (ICLR 2024).", "",
             f"`questions.jsonl`: {n:,} questions, {qf.stat().st_size / 2**20:.1f} MB, sha256 `{sha[:16]}`. "
             f"Length in characters: shortest {L[0]:,}, median {L[n // 2]:,}, longest {L[-1]:,}, "
             f"{sum(L):,} in total.", "",
             "The kind of each question is a keyword rule (`kind_of` in the script; the first rule that matches wins: "
             "code, a request that opens with a writing verb, maths and reasoning, writing, advice, general knowledge). "
             "It is a rough label to spread the sample, not a hand check.", "",
             "| Kind of question | In the sample | Share of the sample | Available after the filters |",
             "|---|--:|--:|--:|"]
    for k in ORDER:
        lines.append(f"| {label[k]} | {kinds[k]:,} | {100 * kinds[k] / n:.0f}% | {pool.get(k, 0):,} |")
    lines += [f"| **All** | **{n:,}** | **100%** | **{sum(pool.values()):,}** |", "",
              "## How many conversations each filter removed", "", "| Step | Conversations |", "|---|--:|",
              f"| Read from the {len(shards)} shards | {drop1.pop('conversations read'):,} |"]
    lines += [f"| Removed: {k} | {v:,} |" for k, v in drop1.items()]
    lines += [f"| Passed the filters | {len(rows):,} |"] + [f"| Removed: {k} | {v:,} |" for k, v in drop2.items()]
    lines += [f"| Left to draw from | {sum(pool.values()):,} |", f"| Drawn | {n:,} |", ""]
    tf.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
