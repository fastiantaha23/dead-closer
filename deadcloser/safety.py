"""Safety layer. Everything here is deterministic code - it does NOT depend on the LLM, so a
prompt-injected or hallucinating model cannot talk its way past it.

Inbound  : opt-out, abuse, legal threats, human requests, "are you a bot?", prompt injection, PII.
Outbound : banned claims, fair-housing steering, invented prices, links/phones, secret/tenant
           leakage, denying being an AI, length. Violations -> template fallback -> silent handoff.
Delivery : quiet hours for proactive messages, per-day frequency cap, opt-out hard stop.
"""
from __future__ import annotations

import re
from datetime import datetime

from .config import local

# ------------------------------------------------------------------ inbound
_OPT_EXACT = {"stop", "stopall", "unsubscribe", "quit", "end", "optout", "opt-out", "parar", "baja", "detener", "arret", "arrêt"}
_OPT_PHRASES = [
    r"\bstop (?:texting|messaging|contacting|emailing|calling|sending|sms|the (?:texts|messages|calls))\b",
    r"\bdo(?: not|n'?t) (?:text|contact|call|message|email|reach out)\b",
    r"\b(?:remove|delete) (?:me|my (?:number|info))\b",
    r"\btake me off\b", r"\bunsubscribe\b", r"\bleave me alone\b", r"\bopt[ -]?out\b",
    r"\bno more (?:texts|messages|calls|emails)\b", r"\bquit (?:texting|messaging)\b",
    r"\bno me (?:escriba|llame|contacte)\b", r"\bstop (?:it|this)\b",
]
_NEGATORS = re.compile(r"(?:don'?t|do not|never|please don'?t|dont)\s+$")

_ABUSE = re.compile(r"\b(?:\w*fuck\w*|\w*shit\w*|bitch\w*|asshole\w*|bastards?|idiots?|morons?|stupid (?:bot|idiots?|people)|dumb (?:bot|idiots?)|piece of crap|scammers?|screw you|jerks?|go to hell)\b", re.I)
_DISTRESS = re.compile(r"\b(?:kill myself|suicid\w*|end my life|want to die|self[- ]harm|hurt myself|i(?:'m| am) in danger|call 911|medical emergency)\b", re.I)
_THREAT = re.compile(r"\b(?:sue you|my (?:lawyer|attorney)|legal action|report you|call the police|cops|fraud|scam(?:ming)?|harass(?:ing|ment)|cease and desist|ftc|tcpa|bbb)\b", re.I)
_HUMAN = re.compile(r"\b(?:get|bring|find|need|want|give) (?:me )?(?:your |a |the |an )?(?:manager|supervisor|human|real person|someone|representative|owner)\b|\b(?:talk|speak|chat|connect)(?: to| with)? (?:a |an |the )?(?:real |actual )?(?:human|person|agent|rep|representative|manager|someone|salesperson|broker)\b|\b(?:real|actual) (?:person|human)\b|\bhave (?:someone|a person|a human|an agent) call me\b|\bcall me\b", re.I)
_BOT_Q = re.compile(r"\b(?:are you|r u|you a|is this) (?:a |an )?(?:bot|robot|human|real person|real|ai|automated|machine|chatbot|artificial)\b|\bam i (?:talking|speaking|chatting) (?:to|with) (?:a )?(?:bot|human|real|robot|person|machine)\b", re.I)
_INJECTION = [
    r"ignore (?:all |any |your |the )?(?:previous|prior|above|earlier)? ?(?:instructions|rules|prompts?|guidelines)",
    r"disregard (?:all |any |your |the )?(?:previous|prior|above)? ?(?:instructions|rules|prompts?)",
    r"(?:reveal|show|print|repeat|output|leak|tell me) (?:me )?(?:your |the )?(?:system|initial|hidden|original) ?(?:prompt|instructions|message)",
    r"repeat (?:everything|all|the (?:text|words|message)|your (?:instructions|prompt)) (?:above|before|prior)", r"print your (?:hidden )?(?:instructions|prompt)",
    r"\bsystem prompt\b", r"\bdeveloper mode\b", r"\bjailbreak\b", r"\bDAN\b", r"you are now (?:a|an|in)\b",
    r"act as (?:a |an )?(?:different|unrestricted|admin|system|developer)", r"pretend (?:you|to) (?:are|be) ",
    r"\bnew instructions?:", r"</?(?:system|assistant|instructions?)>", r"\[\s*(?:system|admin|developer)\s*\]",
    r"override (?:your |the )?(?:rules|policy|policies|safety|pricing|fee)", r"(?:api|secret|admin) ?key",
    r"other (?:customers|clients|tenants|companies|dealers|agencies)'? (?:data|leads|numbers|info|conversations)",
    r"\bsudo\b", r"\bexec(?:ute)?\(", r"\bdrop table\b", r"<script", r"\bprompt injection\b",
]
_INJ_RE = [re.compile(p, re.I) for p in _INJECTION]

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SECRET_WORDS = re.compile(r"\b(?:password|passcode|passwd)\s*(?:is|:|=)\s*\S+|\b(?:cvv|cvc|pin|otp|one[- ]time code)\s*(?:is|:|=)?\s*\d{3,8}\b", re.I)


def _luhn(num: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", num)]
    if not 13 <= len(digits) <= 19:
        return False
    s = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return s % 10 == 0


def find_pii(text: str) -> list[str]:
    found = []
    if _SSN.search(text):
        found.append("ssn")
    if any(_luhn(m.group()) for m in _CARD.finditer(text)):
        found.append("card")
    if _SECRET_WORDS.search(text):
        found.append("credential")
    return found


def redact(text: str) -> str:
    text = _SSN.sub("[redacted-ssn]", text)
    text = _CARD.sub(lambda m: "[redacted-card]" if _luhn(m.group()) else m.group(), text)
    return _SECRET_WORDS.sub("[redacted-credential]", text)


def is_opt_out(text: str) -> bool:
    t = text.strip().lower()
    bare = re.sub(r"[^\w\- ]", "", t).strip()
    if bare in _OPT_EXACT:
        return True
    words = bare.split()
    if words and words[0] in _OPT_EXACT and len(words) <= 4 and "by" not in words[:3]:
        return True
    for p in _OPT_PHRASES:
        for m in re.finditer(p, t):
            if not _NEGATORS.search(t[: m.start()]):
                return True
    return False


def scan_inbound(text: str) -> dict:
    return {
        "opt_out": is_opt_out(text),
        "abuse": bool(_ABUSE.search(text)),
        "threat": bool(_THREAT.search(text)),
        "distress": bool(_DISTRESS.search(text)),
        "human": bool(_HUMAN.search(text)),
        "asks_bot": bool(_BOT_Q.search(text)),
        "injection": any(r.search(text) for r in _INJ_RE),
        "pii": find_pii(text),
        "too_long": len(text) > 2000,
    }


# ------------------------------------------------------------------ outbound
_BANNED = re.compile(
    r"\b(?:guarantee[sd]?|guaranteed|risk[- ]free|no risk|100% (?:safe|sure|approved)|best price|lowest price|price (?:will|is going to) (?:go up|rise|increase)|"
    r"will appreciate|appreciate in value|(?:pre-?)?approved for|you(?:'|’)?re approved|act now or lose|last chance|only one left|going fast|everyone is buying)\b", re.I)
_FAIR_HOUSING = re.compile(
    r"\b(?:safe neighbou?rhood|good neighbou?rhood|bad neighbou?rhood|family[- ]friendly (?:area|neighbou?rhood)|no (?:kids|children)|adults only|"
    r"christian|muslim|jewish|church|mosque|synagogue|white|black|hispanic|asian|ethnic|exclusive (?:area|community)|"
    r"perfect for (?:a )?(?:young|single|married|retired|white|christian)|(?:young|old) (?:couples?|people)|"
    r"mental(?:ly)? (?:ill|health)|disabled|handicap\w*|wheelchair)\b", re.I)
_AI_DENIAL = re.compile(r"\b(?:i(?:'| a)?m (?:a )?(?:real |actual )?(?:human|person|man|woman)|i am (?:a )?(?:real |actual )?(?:human|person)|not (?:a )?(?:bot|ai|robot))\b", re.I)
_LEAK = re.compile(r"(?:system prompt|my instructions|sk-ant|api[_ ]?key|anthropic_api|developer message|as an ai language model)", re.I)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?1[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}(?!\d)")
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|million|thousand)?", re.I)
_PROFANITY = re.compile(r"\b(?:fuck\w*|shit\w*|bitch\w*|asshole|damn|crap)\b", re.I)
MAX_LEN = 480


def parse_money_amounts(text: str) -> set[int]:
    out = set()
    for m in _MONEY.finditer(text):
        v = float(m.group(1).replace(",", ""))
        suf = (m.group(2) or "").lower()
        if suf in ("k", "thousand"):
            v *= 1_000
        elif suf in ("m", "million"):
            v *= 1_000_000
        out.add(int(round(v)))
    return out


def check_outbound(text: str, *, allowed_amounts: set[int], other_tenant_names: list[str],
                   tenant_phone: str = "", vertical: str = "real_estate") -> list[str]:
    v: list[str] = []
    if not text or not text.strip():
        v.append("empty")
        return v
    if len(text) > MAX_LEN:
        v.append("too_long")
    if _BANNED.search(text):
        v.append("banned_claim")
    if vertical == "real_estate" and _FAIR_HOUSING.search(text):
        v.append("fair_housing")
    if _AI_DENIAL.search(text):
        v.append("denies_ai")
    if _LEAK.search(text):
        v.append("leak")
    if _URL.search(text):
        v.append("url")
    for m in _PHONE.finditer(text):
        if not tenant_phone or re.sub(r"\D", "", m.group()) not in re.sub(r"\D", "", tenant_phone):
            v.append("phone_number")
            break
    if _PROFANITY.search(text):
        v.append("profanity")
    low = text.lower()
    for n in other_tenant_names:
        if n and n.lower() in low:
            v.append("cross_tenant_leak")
            break
    bad = [a for a in parse_money_amounts(text) if not any(abs(a - x) <= max(1, int(x * 0.005)) for x in allowed_amounts)]
    if bad:
        v.append("unverified_amount")
    return v


# ------------------------------------------------------------------ delivery gates
def in_send_window(now: datetime, tz_offset_min: int, window: tuple[int, int]) -> bool:
    h = local(now, tz_offset_min).hour
    return window[0] <= h < window[1]


def next_window_open(now: datetime, tz_offset_min: int, window: tuple[int, int]) -> datetime:
    """Earliest time >= now that is inside the send window (opens at window[0]+1h to feel natural)."""
    from datetime import timedelta
    l = local(now, tz_offset_min)
    open_h = window[0] + 1
    if window[0] <= l.hour < window[1]:
        return now
    target = l.replace(hour=open_h, minute=0, second=0, microsecond=0)
    if l.hour >= window[1] or target <= l:
        target += timedelta(days=1)
    return target.astimezone(now.tzinfo)
