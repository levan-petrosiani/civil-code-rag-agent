import re
from typing import List, Dict

def chunk_georgian_civil_code(full_text: str) -> List[Dict]:
    MAX_CHUNK_SIZE = 1200  # smaller chunks improve recall
    if not full_text:
        return []

    # Split by article
    article_pattern = r'(?=მუხლი [\d\u200b¹²³⁴⁵⁶⁷⁸⁹⁰]+\.?\s*.*)'
    raw_articles = re.split(article_pattern, full_text)

    chunks = []
    current_book = None
    current_chapter = None

    for article_text in raw_articles:
        if not article_text.strip():
            continue

        # detect book/chapter markers
        book_match = re.search(r'წიგნი\s+[^\s]+', article_text)
        if book_match:
            current_book = book_match.group(0)

        chapter_match = re.search(r'თავი\s+[^\s]+', article_text)
        if chapter_match:
            current_chapter = chapter_match.group(0)

        # parse article header
        article_header_match = re.search(
            r'მუხლი ([\d\u200b¹²³⁴⁵⁶⁷⁸⁹⁰]+)\.?\s*(.*)', article_text
        )
        if not article_header_match:
            continue

        article_number_str = article_header_match.group(1).replace('\u200b', '')
        article_title = article_header_match.group(2).strip()

        # remove the "მუხლი N ..." line from text
        cleaned_text = re.sub(
            r'მუხლი [\d\u200b¹²³⁴⁵⁶⁷⁸⁹⁰]+\.?\s*.*\n?', '', article_text, 1
        ).strip()

        base_metadata = {
            "source": "საქართველოს სამოქალაქო კოდექსი",
            "book": current_book if current_book else "უცნობი წიგნი",
            "chapter": current_chapter if current_chapter else "უცნობი თავი",
            "article_number": article_number_str,
            "article_title": article_title
        }


        # Keep chunk text simple & focused
        header_str = f"მუხლი {article_number_str}. {article_title}"

        # If article fits in one chunk
        if len(cleaned_text) <= MAX_CHUNK_SIZE:
            chunks.append({
                "text": f"{header_str}\n\n{cleaned_text}",
                "metadata": base_metadata
            })
        else:
            # Split by numbered paragraphs (1. 2. ა) ბ) etc.)
            paragraph_pattern = r'(?=\d+\.\s|[ა-ი]\)\s)'
            sub_chunks = re.split(paragraph_pattern, cleaned_text)

            for i, sub_chunk_text in enumerate(sub_chunks):
                if sub_chunk_text.strip():
                    sub_chunk_metadata = base_metadata.copy()
                    sub_chunk_metadata['sub_chunk_seq'] = i + 1

                    chunk_text = (
                        f"{header_str} (ნაწილი {i+1})\n\n{sub_chunk_text.strip()}"
                    )
                    chunks.append({
                        "text": chunk_text,
                        "metadata": sub_chunk_metadata
                    })

    return chunks
