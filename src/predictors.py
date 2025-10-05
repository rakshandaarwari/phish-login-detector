import re
import pandas as pd
import tldextract
from transformers import pipeline

URL_RE = re.compile(r'https?://[^\s"<>]+')


def extract_urls(text):
    if not isinstance(text, str):
        return []
    return URL_RE.findall(text)


def has_punycode(urls):
    return any('xn--' in u for u in urls)


def url_with_ip(urls):
    ip_re = re.compile(r'https?://\d+\.\d+\.\d+\.\d+')
    return any(ip_re.search(u) for u in urls)


def suspicious_url_pattern(urls):
    score = 0
    for u in urls:
        if 'xn--' in u:
            score += 1
        if url_with_ip([u]):
            score += 1
        if u.count('-') > 3:
            score += 1
        if len(u) > 120:
            score += 1
        if '?' in u and len(u.split('?')[-1]) > 40:
            score += 1
    return score


def header_mismatch(from_addr, reply_to):
    if not from_addr or not reply_to:
        return 0
    try:
        fdom = from_addr.split('@')[-1].lower()
        rdom = reply_to.split('@')[-1].lower()
        return int(fdom != rdom)
    except:
        return 0


_zero_shot = None
def get_zero_shot():
    global _zero_shot
    if _zero_shot is None:
        _zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _zero_shot


def zero_shot_score(text):
    clf = get_zero_shot()
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    out = clf(text, candidate_labels=["phishing", "benign"], hypothesis_template="This text is {}.")
    return float(dict(zip(out['labels'], out['scores'])).get('phishing', 0.0))


def predict_email_risk(subject, body, from_addr="", reply_to=""):
    text = (subject or "") + " " + (body or "")
    urls = extract_urls(text)
    feat = {
        'num_urls': len(urls),
        'punycode': int(has_punycode(urls)),
        'ip_url': int(url_with_ip(urls)),
        'suspicious_url_score': suspicious_url_pattern(urls),
        'header_mismatch': header_mismatch(from_addr, reply_to),
        'has_form': int('<form' in str(body).lower()),
        'text_score': zero_shot_score(text)
    }
    weights = [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2]
    risk = sum(feat[k] * w for k, w in zip(feat.keys(), weights))
    risk = min(1.0, max(0.0, risk))
    return {'risk': risk, 'features': feat}


def login_rule_risk(row, usual_country_map):
    risk = 0.0
    if usual_country_map.get(row['user']) and row['country'] != usual_country_map[row['user']]:
        risk += 0.6
    if row.get('failed_prior', 0) >= 2:
        risk += 0.3
    if 'device' in row and row['device'] not in ['pc-1', 'pc-2', 'phone-1', 'phone-2']:
        risk += 0.2
    return min(1.0, risk)
