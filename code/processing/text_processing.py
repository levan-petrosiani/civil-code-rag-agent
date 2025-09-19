import re

def get_paragraph_text_with_superscripts(para):
    superscripts = {str(i): c for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")}
    text_parts = []
    for run in para.runs:
        if run.font.superscript:
            text_parts.append("".join(superscripts.get(ch, ch) for ch in run.text))
        else:
            text_parts.append(run.text)
    return "".join(text_parts)


def clean_noise(document):
    NOISE_PATTERN = re.compile(
        r"(საქართველოს\s+საკონსტიტუციო\s+სასამართლოს\s+\d{4}\s+წლის\s+\d{1,2}\s+[ა-ჰ]+\s+გადაწყვეტილება\s+№[\d/,]+\s*–?\s*-?\s*(?:სსმ|ვებგვერდი).*?(?:\n|$))"
        r"|"
        r"(საქართველოს\s+\d{4}\s+წლის\s*\d{1,2}\s+[ა-ჰ]+\s+კანონი\s+№\d+\s*–?\s*-?\s*(?:სსმ|ვებგვერდი).*?(?:\n|$))",
        flags=re.DOTALL
    )

    cleaned = []
    for para in document.paragraphs:
        text = get_paragraph_text_with_superscripts(para)
        if NOISE_PATTERN.search(text):
            text = NOISE_PATTERN.sub("", text).strip()
        cleaned.append(text)

    if cleaned:
        cleaned[-1] = ""  # remove last noise
    return cleaned
