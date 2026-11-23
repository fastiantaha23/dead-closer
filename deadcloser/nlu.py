"""Rule-based natural-language understanding for inbound lead messages.

This is the offline "brain" used when no LLM key is configured and the safety net when the
LLM fails. Output schema (identical to the Claude tool schema in llm.py):

  intents       set of: accept decline book_request reschedule cancel_visit not_interested wrong_number
                has_agent objection_price objection_timing need_partner ask_price ask_availability
                ask_location ask_features ask_financing ask_options greet question
  budget_min/max, budget_monthly, budget_ambiguous, timeline_days, timeline_text, browsing
  prefs         bedrooms, body, fuel, brand, area, features[]
  slot          date (YYYY-MM-DD), hour, part, index, weekend
  item_id       inventory item the lead refers to (if unambiguous)
  name          self-introduced name
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from .config import local

WORD_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "twelve": 12, "couple": 2, "few": 3, "several": 4}
MONTHS = {"jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
          "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
          "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12}
SEASONS = {"spring": (4, 1), "summer": (6, 21), "fall": (9, 22), "autumn": (9, 22), "winter": (12, 21)}
WEEKDAYS = {"monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3,
            "thur": 3, "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6}
UNIT_AFTER = {"bed", "beds", "bedroom", "bedrooms", "bhk", "br", "bath", "baths", "bathroom", "bathrooms", "day", "days",
              "week", "weeks", "month", "months", "year", "years", "yr", "yrs", "am", "pm", "min", "mins", "minute",
              "minutes", "hour", "hours", "hr", "hrs", "mile", "miles", "mi", "km", "sqft", "sq", "ft", "st", "nd", "rd",
              "th", "people", "kids", "seats", "doors", "cars", "of", "pm.", "am."}
SUFFIX = {"k": 1_000, "thousand": 1_000, "grand": 1_000, "m": 1_000_000, "mm": 1_000_000, "mil": 1_000_000,
          "million": 1_000_000, "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "cr": 10_000_000, "crore": 10_000_000}

MONEY_RE = re.compile(
    r"(?P<cur>\$)?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<suf>thousand|million|lakhs?|crore|grand|mil|mm|lac|cr|k|m)?(?![a-z0-9])",
    re.I)
MAX_CTX = re.compile(r"(?:under|below|max(?:imum)?|up to|at most|no more than|less than|not (?:to )?exceed|cap(?:ped)? at|within|ceiling|budget (?:of|is)|around|about|approx\w*|roughly|~|upto)\s*$", re.I)
MIN_CTX = re.compile(r"(?:over|above|at least|minimum|more than|starting (?:at|from)|from)\s*$", re.I)
MONTHLY_RE = re.compile(r"(?:/\s?mo\b|/\s?month|per month|a month|each month|monthly|every month|\bemi\b|/\s?m\b|\bpm\b)", re.I)

BROWSING_RE = re.compile(r"\b(?:just (?:browsing|looking|curious|checking)|no rush|not in (?:a )?(?:hurry|rush)|no hurry|exploring|early stages?|window[- ]shopping|someday|eventually|no (?:firm |set )?timeline|not sure when|don'?t know when|haven'?t decided when|whenever|just researching)\b", re.I)

ACCEPT_RE = re.compile(r"^\W*(?:yes|yep|yeah|yup|yea|sure|ok(?:ay)?|k|sounds good|works(?: for me)?|that works|perfect|great|definitely|let'?s do it|book it|confirm(?:ed)?|i'?ll take it|absolutely|go ahead|please do|that'?s fine|fine|good|alright|sounds great|do it|y)\b|\b(?:that works|works for me|sounds good|book (?:it|me|that)|i'?ll (?:be there|take)|see you then|let'?s do)\b", re.I)
DECLINE_RE = re.compile(r"^\W*(?:no|nope|nah|none of (?:those|them)|neither|not (?:really|those|that)|can'?t (?:make|do)|doesn'?t work|those don'?t work)\b|\b(?:none (?:of those|work)|don'?t work for me|different (?:time|day)|another (?:time|day))\b", re.I)
BOOK_RE = re.compile(r"\b(?:book|schedule|set up|arrange|see (?:it|the|this|them|one)|view(?:ing)?|visit|tour|walk[- ]?through|show(?:ing)? me|come (?:by|see|in|over)|test[- ]?drive|drive (?:it|one)|take (?:it|one) for a (?:spin|drive)|stop by|swing by|check (?:it|them) out|available to (?:see|view)|when can i|open house|appointment|i(?:'d| would) like to (?:see|come))\b", re.I)
RESCHEDULE_RE = re.compile(r"\b(?:resched\w*|change (?:the |my )?(?:time|date|appointment|visit|booking)|move (?:it|the|my)|push (?:it|back)|can we do (?:another|a different)|another (?:time|day))\b", re.I)
CANCEL_RE = re.compile(r"\b(?:cancel(?:l?ed|ling)?|call (?:it )?off|can'?t make it|won'?t (?:make|be able)|no longer (?:coming|need))\b", re.I)
NOT_INT_RE = re.compile(r"\b(?:not interested|no longer (?:interested|looking|searching|in the market)|no thanks|no thank you|already (?:bought|purchased|found|got|closed|signed|have)|found (?:a |another |something |one |an? )|went with (?:another|someone|a different)|bought (?:one|elsewhere|something|a )|changed my mind|don'?t need (?:it|anything|one) anymore|not looking|not buying|pass on|(?:buy|buying|bought|purchase\w*|going|went|go|decided)\s+(?:to\s+)?(?:\w+\s+)?(?:elsewhere|somewhere else)|decided against)\b", re.I)
WRONG_RE = re.compile(r"\b(?:wrong number|who is this|never (?:signed|asked|inquired|filled|requested)|didn'?t (?:sign|ask|inquire|fill|request))\b", re.I)
AGENT_RE = re.compile(r"\b(?:(?:already )?(?:working|have|using|signed|represented) (?:with|by) (?:an? )?(?:agent|realtor|broker|dealer|dealership|salesperson|someone)|my (?:agent|realtor|broker))\b", re.I)
PRICE_OBJ_RE = re.compile(r"\b(?:too (?:expensive|high|much|pricey)|over (?:my )?budget|out of (?:my )?(?:budget|range|price)|can'?t afford|pricey|(?:a )?(?:bit|little|lot|tad) (?:much|high|steep|pricey)|(?:price|cost) is (?:high|steep|too)|cheaper|lower (?:price|cost)|discount|negotiat\w*|best (?:you can do|deal)|any (?:deals|offers|incentives)|can you do better|better price)\b", re.I)
TIMING_RE = re.compile(r"\b(?:not (?:right )?now|not yet|bad time|too busy|busy (?:this|right|these)|later on|check back|call me (?:in|next|later|after)|reach out (?:in|next|later|after)|text me (?:in|next|later|after)|down the (?:road|line)|in the future|maybe (?:later|next)|come back to me|after (?:the )?(?:holidays|summer|new year))\b", re.I)
PARTNER_RE = re.compile(r"\b(?:my (?:wife|husband|spouse|partner|girlfriend|boyfriend|fianc\w+|parents|mom|dad|family|roommate)|(?:talk|check|ask|discuss)\w* (?:with|to) (?:my )?(?:wife|husband|spouse|partner|family)|we(?:'ll| will)? (?:need|have) to (?:discuss|talk))\b", re.I)
Q_PRICE_RE = re.compile(r"\b(?:how much|price|cost|asking|what(?:'s| is) (?:it|the) (?:going )?for|msrp|list(?:ing)? price|pricing|out the door|what does it go for)\b", re.I)
Q_AVAIL_RE = re.compile(r"\b(?:still available|is (?:it|this|that|the \w+) (?:still )?(?:available|for sale|on the market|in stock)|in stock|still (?:there|for sale|on the market)|has it sold|been sold)\b", re.I)
Q_LOC_RE = re.compile(r"\b(?:where|address|location|directions|which (?:area|neighbou?rhood|city)|how far|located)\b", re.I)
Q_FEAT_RE = re.compile(r"\b(?:bed(?:room)?s?|bath(?:room)?s?|sq(?:uare)? ?(?:ft|feet)|sqft|mileage|miles on|trim|colou?r|pool|garage|yard|hoa|pets?|schools?|taxes|utilities|mpg|range|battery|warranty|carfax|accident|features?|specs?|include[sd]?|does it (?:have|come)|is there a|how many|how big|size)\b", re.I)
Q_FIN_RE = re.compile(r"\b(?:financ\w*|mortgage|loan|down ?payment|emi|interest rate|apr|lease|trade[- ]?in|credit (?:score|check)|monthly payment|pre-?approv\w*|incentives?|rebates?)\b", re.I)
DEMOGRAPHICS_RE = re.compile(r"\b(?:safe(?:ty)?|crime|dangerous|rough|sketchy|ghetto|demographic\w*|diverse|diversity|immigrants?|race|racial|religio\w*|church|mosque|synagogue|temple|kids|children|families|singles|seniors|elderly|young people|who lives|what kind of (?:people|neighbou?rs)|good (?:area|neighbou?rhood|schools)|bad (?:area|neighbou?rhood)|white|black|asian|hispanic|latino|muslim|christian|jewish|gay|disabled)\b", re.I)
PROMISE_RE = re.compile(r"\b(?:guarantee\w*|promise|will (?:the )?(?:price|value|it) (?:go|drop|rise|appreciate|increase|decrease)|appreciate in value|good investment|will i (?:get|be) approved|assure|swear)\b", re.I)
OPTIONS_RE = re.compile(r"\b(?:other (?:options|ones|listings|cars|vehicles|homes|properties)|more (?:options|like)|similar|alternatives?|anything else|something else|what else|any others|other places)\b", re.I)
GREET_RE = re.compile(r"^\W*(?:hi|hello|hey|hola|good (?:morning|afternoon|evening)|yo|sup|howdy)\b", re.I)
NAME_RE = re.compile(r"\b(?i:i'?m|i am|this is|my name is|name'?s)\s+([A-Z][a-z]{1,15})\b")

BODY_RE = re.compile(r"\b(suv|sedan|truck|pickup|hatchback|coupe|minivan|van|crossover|convertible)\b", re.I)
FUEL_RE = re.compile(r"\b(hybrid|electric|ev|diesel|plug-?in)\b", re.I)
BED_RE = re.compile(r"\b(\d|one|two|three|four|five|six)[\s-]*(?:bed(?:room)?s?|bhk|br)\b", re.I)
FEATURE_VOCAB = ["pool", "garage", "yard", "garden", "balcony", "gym", "patio", "office", "porch", "rooftop", "parking",
                 "awd", "4x4", "sunroof", "leather", "third row", "tow package", "carplay", "autopilot", "smart home"]


# ---------------------------------------------------------------------------- money
def parse_budget(text: str, awaiting: str | None, monthly_factor: int = 60) -> dict:
    out = {"budget_min": None, "budget_max": None, "budget_monthly": False, "budget_ambiguous": False}
    cands = []
    for m in MONEY_RE.finditer(text):
        raw = m.group("num")
        after = text[m.end():m.end() + 12].strip().lower()
        nxt = re.match(r"[a-z\.]+", after)
        if nxt and nxt.group().rstrip(".") in UNIT_AFTER and not m.group("cur") and not m.group("suf"):
            continue
        # skip times (3:30), dates (11/28), phone-ish digit runs, years
        pre = text[max(0, m.start() - 1):m.start()]
        post = text[m.end():m.end() + 1]
        if (post == ":" and re.match(r"\d", text[m.end() + 1:m.end() + 2] or "")) or post == "/" and re.match(r"\d", text[m.end() + 1:m.end() + 2] or ""):
            continue
        if pre == ":" or pre == "/":
            continue
        digits = re.sub(r"\D", "", raw)
        if len(digits) >= 10 and "," not in raw:
            continue
        val = float(raw.replace(",", ""))
        suf = (m.group("suf") or "").lower()
        mult = SUFFIX.get(suf, 1)
        has_signal = bool(m.group("cur") or suf)
        value = val * mult
        if value < 100:          # "$1", "5" - never a plausible budget
            continue
        if not has_signal:
            if 1990 <= value <= 2035 and awaiting != "budget":
                continue
            if value >= 10_000:
                has_signal = True
            elif awaiting == "budget" and value >= 100:
                has_signal = True
            elif re.search(r"budget|afford|spend|pay|under|around|up to|max", text[max(0, m.start() - 25):m.start()], re.I) and value >= 100:
                has_signal = True
        if not has_signal:
            tail = text[m.end():]
            rng = re.match(r"\s*(?:-|–|—|to|and|through)\s*\$?\s*\d[\d,\.]*\s*(?:k|m|thousand|million|grand)\b", tail, re.I)
            if rng and value >= 100:
                has_signal = True
        if not has_signal:
            continue
        cands.append((m, value, bool(suf), bool(m.group("cur"))))
    if not cands:
        return out
    monthly = bool(MONTHLY_RE.search(text))
    out["budget_monthly"] = monthly
    # range?
    if len(cands) >= 2:
        (m1, v1, s1, _), (m2, v2, s2, _) = cands[0], cands[1]
        between = text[m1.end():m2.start()]
        if re.fullmatch(r"\s*(?:-|–|—|to|and|through|or)\s*\$?\s*", between, re.I):
            if not s1 and s2 and v1 < v2 and v1 < 10_000:
                for suf_val in (1_000, 1_000_000):
                    if abs(v1 * suf_val - v2) < v2 and v1 * suf_val <= v2:
                        v1 = v1 * (SUFFIX[(m2.group("suf") or "k").lower()])
                        break
            lo, hi = sorted((v1, v2))
            out["budget_min"], out["budget_max"] = int(lo), int(hi)
            return _finish_budget(out, monthly, monthly_factor)
    m, v, suf_flag, cur_flag = cands[0]
    before = text[max(0, m.start() - 30):m.start()]
    if MIN_CTX.search(before) and not MAX_CTX.search(before):
        out["budget_min"] = int(v)
    else:
        out["budget_max"] = int(v)
    if not suf_flag and not cur_flag and v < 10_000 and not monthly:
        out["budget_ambiguous"] = True
    return _finish_budget(out, monthly, monthly_factor)


def _finish_budget(out: dict, monthly: bool, monthly_factor: int) -> dict:
    if monthly:
        # keep the monthly figure but expose an approximate total for inventory matching
        for k in ("budget_min", "budget_max"):
            if out[k]:
                out[k] = int(out[k] * monthly_factor)
    return out


# ---------------------------------------------------------------------------- timeline
def parse_timeline(text: str, now: datetime) -> tuple[int | None, str | None, bool]:
    t = text.lower()
    if BROWSING_RE.search(t):
        return 180, "no firm timeline", True
    m = re.search(r"\b(?:in|within|about|around|over the next|next|after|for the next)?\s*(\d{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|twelve|couple(?: of)?|few|several)\s*(?:more )?(days?|weeks?|months?|years?)\b", t)
    if m:
        qty_raw = m.group(1).replace(" of", "")
        qty = int(qty_raw) if qty_raw.isdigit() else WORD_NUM.get(qty_raw, 1)
        unit = m.group(2)
        mult = 1 if unit.startswith("day") else 7 if unit.startswith("week") else 30 if unit.startswith("month") else 365
        days = qty * mult
        return days, f"within ~{qty} {unit.rstrip('s')}{'s' if qty != 1 else ''}", False
    fixed = [
        (r"\b(?:asap|immediately|right away|urgent\w*|as soon as (?:possible|i can)|today|tonight|right now|this very week|(?:need|want) (?:it|one|something|a \w+) now)\b", 2, "immediately"),
        (r"\b(?:this week|tomorrow|this weekend|next few days|couple of days)\b", 5, "this week"),
        (r"\bnext week\b", 10, "next week"),
        (r"\b(?:this month|end of (?:the )?month|few weeks)\b", 20, "this month"),
        (r"\bnext month\b", 35, "next month"),
        (r"\b(?:end of (?:the )?year|by year[- ]end|this year|before (?:the )?year ends?)\b", None, "by year end"),
        (r"\bnext year\b", 300, "next year"),
    ]
    for pat, days, label in fixed:
        if re.search(pat, t):
            if days is None:
                end = datetime(now.year, 12, 31, tzinfo=now.tzinfo)
                days = max(7, (end - now).days)
            return days, label, False
    mm = re.search(r"\b(?:in|by|before|around|until|early|late|mid|end of|for|this|next|come|starting)\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\b", t)
    if mm:
        month = MONTHS[mm.group(1)]
        year = now.year
        target = datetime(year, month, 15, tzinfo=now.tzinfo)
        if target <= now:
            target = datetime(year + 1, month, 15, tzinfo=now.tzinfo)
        return max(3, (target - now).days), f"around {mm.group(1).capitalize()}", False
    sm = re.search(r"\b(?:this|next|by|in the|in)\s+(spring|summer|fall|autumn|winter)\b", t)
    if sm:
        mo, dy = SEASONS[sm.group(1)]
        target = datetime(now.year, mo, dy, tzinfo=now.tzinfo)
        if target <= now:
            target = datetime(now.year + 1, mo, dy, tzinfo=now.tzinfo)
        return max(3, (target - now).days), f"by {sm.group(1)}", False
    return None, None, False


# ---------------------------------------------------------------------------- slot preference
def parse_slot_pref(text: str, now: datetime, tz_min: int, awaiting: str | None) -> dict:
    t = text.lower()
    lnow = local(now, tz_min)
    pref = {"date": None, "hour": None, "part": None, "index": None, "weekend": False}
    if awaiting in ("slot", "confirm_slot"):
        m = re.fullmatch(r"\W*(?:option|number|#|choice)?\s*([1-3])(?:st|nd|rd)?\W*", t)
        if m:
            pref["index"] = int(m.group(1))
    if pref["index"] is None:
        m = re.search(r"\b(?:option|number|choice|#)\s*([1-3])\b", t)
        if m:
            pref["index"] = int(m.group(1))
        elif re.search(r"\b(?:the )?(?:first|1st)(?: one| option| slot)?\b", t) and awaiting in ("slot", "confirm_slot"):
            pref["index"] = 1
        elif re.search(r"\b(?:the )?(?:second|2nd)(?: one| option| slot)?\b", t) and awaiting in ("slot", "confirm_slot"):
            pref["index"] = 2
        elif re.search(r"\b(?:the )?(?:third|3rd|last)(?: one| option| slot)?\b", t) and awaiting in ("slot", "confirm_slot"):
            pref["index"] = 3
    date = None
    if re.search(r"\bday after tomorrow\b", t):
        date = lnow.date() + timedelta(days=2)
    elif re.search(r"\btomorrow\b", t):
        date = lnow.date() + timedelta(days=1)
    elif re.search(r"\b(?:today|tonight|this evening|this afternoon|this morning)\b", t):
        date = lnow.date()
    wd = None
    for name, idx in WEEKDAYS.items():
        if re.search(rf"\b{name}\b", t):
            wd = idx
            break
    if date is None and wd is not None:
        delta = (wd - lnow.weekday()) % 7
        if delta == 0 and not re.search(r"\btoday\b", t):
            delta = 7
        if re.search(r"\bnext\s+" + [n for n, i in WEEKDAYS.items() if i == wd][0] + r"\b", t) and delta < 7:
            pass
        date = lnow.date() + timedelta(days=delta)
    if date is None and re.search(r"\b(?:this )?weekend\b", t):
        delta = (5 - lnow.weekday()) % 7
        date = lnow.date() + timedelta(days=delta)
        pref["weekend"] = True
    if date is None:
        m = re.search(r"\b(?:on )?(?:the )?(\d{1,2})(?:st|nd|rd|th)\b", t)
        if m and 1 <= int(m.group(1)) <= 31:
            d = int(m.group(1))
            try:
                cand = lnow.date().replace(day=d)
                if cand < lnow.date():
                    mth = lnow.month % 12 + 1
                    yr = lnow.year + (1 if lnow.month == 12 else 0)
                    cand = cand.replace(year=yr, month=mth)
                date = cand
            except ValueError:
                pass
    if date is None:
        m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", t)
        if m:
            try:
                cand = lnow.date().replace(month=int(m.group(1)), day=int(m.group(2)))
                if cand < lnow.date():
                    cand = cand.replace(year=cand.year + 1)
                date = cand
            except ValueError:
                pass
    if date is None and re.search(r"\bnext week\b", t):
        date = lnow.date() + timedelta(days=(7 - lnow.weekday()))
    if date:
        pref["date"] = date.isoformat()
    # time of day
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", t)
    if m:
        h = int(m.group(1)) % 12
        if m.group(3).startswith("p"):
            h += 12
        pref["hour"] = h
    else:
        m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
        if m and 0 <= int(m.group(1)) <= 23:
            pref["hour"] = int(m.group(1))
        else:
            m = re.search(r"\b(?:at|around|by|@)\s*(\d{1,2})\b(?!\s*(?:bed|bath|k\b|,))", t)
            if m and 1 <= int(m.group(1)) <= 12 and pref["index"] is None:
                h = int(m.group(1))
                pref["hour"] = h + 12 if 1 <= h <= 6 else h
    if pref["hour"] is None:
        if re.search(r"\bnoon\b|\blunch\b", t):
            pref["hour"] = 12
        elif re.search(r"\bafter work\b", t):
            pref["hour"] = 17
        elif re.search(r"\bmorning\b", t):
            pref["part"] = "morning"
        elif re.search(r"\bafternoon\b", t):
            pref["part"] = "afternoon"
        elif re.search(r"\bevening\b", t):
            pref["part"] = "evening"
    return pref


# ---------------------------------------------------------------------------- preferences
def parse_prefs(text: str, ctx: dict) -> dict:
    prefs: dict = {}
    m = BED_RE.search(text)
    if m:
        g = m.group(1).lower()
        prefs["bedrooms"] = int(g) if g.isdigit() else WORD_NUM.get(g)
    elif re.search(r"\bstudio\b", text, re.I):
        prefs["bedrooms"] = 0
    m = BODY_RE.search(text)
    if m:
        b = m.group(1).lower()
        prefs["body"] = {"pickup": "truck", "crossover": "suv"}.get(b, b)
    m = FUEL_RE.search(text)
    if m:
        f = m.group(1).lower()
        prefs["fuel"] = "electric" if f == "ev" else f
    low = text.lower()
    for brand in ctx.get("brands", []):
        if brand.lower() in low:
            prefs["brand"] = brand
            break
    for area in ctx.get("areas", []):
        if area.lower() in low:
            prefs["area"] = area
            break
    feats = [f for f in FEATURE_VOCAB if re.search(rf"\b{re.escape(f)}\b", low)]
    if feats:
        prefs["features"] = feats
    return prefs


def find_item(text: str, items: list[dict]) -> str | None:
    low = text.lower()
    hits = []
    generic = {"home", "house", "condo", "bed", "with", "new", "build", "family", "view", "downtown", "the", "and"}
    for it in items:
        toks = {w for w in re.findall(r"[a-z0-9\-]+", it["title"].lower()) if len(w) >= 4 and w not in generic and not w.isdigit()}
        toks.add(it["id"].lower())
        area = (it["attrs"].get("area") or "").lower()
        score = 0
        for tk in toks:
            if re.search(rf"\b{re.escape(tk)}\b", low):
                score += 2 if tk == it["id"].lower() else 1
        if area and area in low:
            score += 1
        if score:
            hits.append((score, it["id"]))
    if not hits:
        return None
    hits.sort(reverse=True)
    if len(hits) > 1 and hits[0][0] == hits[1][0]:
        return None
    return hits[0][1]


# ---------------------------------------------------------------------------- main entry
def understand(text: str, ctx: dict) -> dict:
    now: datetime = ctx["now"]
    awaiting = ctx.get("awaiting")
    tz = ctx.get("tz_offset_min", 0)
    t = text.strip()
    low = t.lower()
    intents: set[str] = set()

    budget = parse_budget(t, awaiting, ctx.get("monthly_factor", 60))
    days, ttext, browsing = parse_timeline(t, now)
    slot = parse_slot_pref(t, now, tz, awaiting)
    prefs = parse_prefs(t, ctx)

    # A phrase like "call me in 2 months" is a timing objection with a timeline attached.
    if ACCEPT_RE.search(t) and not re.search(r"\b(?:but|however|except)\b", low):
        intents.add("accept")
    if DECLINE_RE.search(t):
        intents.add("decline")
        intents.discard("accept")
    if BOOK_RE.search(t):
        intents.add("book_request")
    if RESCHEDULE_RE.search(t):
        intents.add("reschedule")
    if CANCEL_RE.search(t):
        intents.add("cancel_visit")
    if WRONG_RE.search(t):
        intents.add("wrong_number")
        intents.add("not_interested")
    if NOT_INT_RE.search(t):
        intents.add("not_interested")
    if AGENT_RE.search(t):
        intents.add("has_agent")
    if PRICE_OBJ_RE.search(t):
        intents.add("objection_price")
    if TIMING_RE.search(t):
        intents.add("objection_timing")
    if PARTNER_RE.search(t):
        intents.add("need_partner")
    if Q_PRICE_RE.search(t) and "objection_price" not in intents:
        intents.add("ask_price")
    if Q_AVAIL_RE.search(t):
        intents.add("ask_availability")
    if Q_LOC_RE.search(t):
        intents.add("ask_location")
    if Q_FIN_RE.search(t):
        intents.add("ask_financing")
    if Q_FEAT_RE.search(t) and "?" in t:
        intents.add("ask_features")
    if ctx.get("vertical") == "real_estate" and DEMOGRAPHICS_RE.search(t) and ("?" in t or re.search(r"\b(?:who lives|what kind of|tell me about the (?:area|neighbou?rhood))\b", low)):
        intents.add("ask_demographics")
    if PROMISE_RE.search(t):
        intents.add("ask_promise")
    if OPTIONS_RE.search(t):
        intents.add("ask_options")
    if GREET_RE.search(t):
        intents.add("greet")
    if "?" in t and not (intents - {"greet", "accept"}):
        intents.add("question")

    # context-sensitive: bare answers to the question we just asked
    if awaiting in ("slot", "confirm_slot") and slot["date"]:
        days, ttext = None, None  # "tomorrow afternoon" answers the slot question, it is not a buying timeline
    if awaiting == "timeline" and days is None and not browsing and re.search(r"\b(?:not sure|don'?t know|unsure|no idea|flexible)\b", low):
        browsing, days, ttext = True, 180, "flexible"
    if awaiting == "slot" and slot["index"] is None and "accept" in intents and len(ctx.get("offered", [])) == 1:
        slot["index"] = 1

    # "Yes" while we are waiting for a budget/timeline answer is not a booking acceptance
    if awaiting in ("budget", "timeline") and "accept" in intents and not (budget["budget_max"] or budget["budget_min"] or days):
        intents.discard("accept")

    name = None
    m = NAME_RE.search(t)
    if m and m.group(1).lower() not in ("interested", "looking", "just", "not", "sorry", "fine", "good", "ok", "okay", "sure", "ready", "here", "from", "also", "still", "so", "very", "calling", "writing", "texting", "trying", "wondering", "about", "back", "available", "able", "going", "happy", "glad", "thinking"):
        name = m.group(1)

    return {
        "intents": sorted(intents),
        **budget,
        "timeline_days": days, "timeline_text": ttext, "browsing": browsing,
        "prefs": prefs, "slot": slot, "item_id": find_item(t, ctx.get("items", [])), "name": name,
        "sentiment": _sentiment(low),
    }


def _sentiment(low: str) -> float:
    pos = len(re.findall(r"\b(?:love|great|awesome|excited|perfect|thanks|thank you|amazing|nice|interested|good|looking forward)\b", low))
    neg = len(re.findall(r"\b(?:hate|annoy\w*|angry|upset|terrible|awful|worst|waste|ridiculous|disappointed|frustrat\w*|spam)\b", low))
    return max(-1.0, min(1.0, (pos - neg) / 3))
