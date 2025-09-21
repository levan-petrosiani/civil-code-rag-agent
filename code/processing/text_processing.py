import re

def get_paragraph_text_with_superscripts(para):
    """
    ეს ფუნქცია იღებს დოკუმენტის პარაგრაფს და ამუშავებს მის ტექსტს. 
    ის ამოწმებს თითოეულ ტექსტურ ფრაგმენტს (run) პარაგრაფში და თუ ფრაგმენტი ზედა ინდექსითაა (superscript), 
    იცვლება შესაბამისი ზედა ინდექსის სიმბოლოებით (მაგ., 0 ხდება ⁰). საბოლოოდ აბრუნებს პარაგრაფის სრულ ტექსტს, სადაც ზედა ინდექსები სწორადაა წარმოდგენილი.
    """

    superscripts = {str(i): c for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")}
    text_parts = []
    for run in para.runs:
        if run.font.superscript:
            text_parts.append("".join(superscripts.get(ch, ch) for ch in run.text))
        else:
            text_parts.append(run.text)
    return "".join(text_parts)


def clean_noise(document):
    """
    ფუნქცია იღებს დოკუმენტს და ასუფთავებს მას "ხმაურისგან" (noise) ტექსტისაგან, 
    როგორიცაა საქართველოს საკონსტიტუციო სასამართლოს გადაწყვეტილებების ან კანონების ციტირებები (მაგ., „საქართველოს 2024 წლის გადაწყვეტილება №123...“). 
    ის იყენებს რეგულარულ გამოსახულებას (NOISE_PATTERN) ამგვარი ნაწილების იდენტიფიცირებისთვის,
    შლის მათ და აბრუნებს გასუფთავებული პარაგრაფების სიას. ბოლო პარაგრაფი ცარიელდება, რათა თავიდან აიცილოს ნარჩენი ხმაური.
    """
    
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
