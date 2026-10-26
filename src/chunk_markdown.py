import re
def split_into_sections(text):
    heading_pattern = re.compile(
        r"(?m)^(#{1,2})\s+(.+?)\s*$"
    )
    matches = list(heading_pattern.finditer(text))
    sections = []

    if matches and text[:matches[0].start()].strip():
        sections.append(
            {
                "section": "Preamble",
                "text": text[:matches[0].start].strip()
            }
        )
    if not matches:
        return[{
            "section": "Document",
            "text": text.strip
        }]
    for i, match in enumerate(matches):
        start = match.start()
        end = (matches[i+1].start() if i+1 < len(matches) else len(text))
        heading = match.group(2).strip()
        section_text = text[start:end].strip()
        sections.append({
            "section": heading,
            "text": section_text
        })
    return sections

def split_into_sentences(text):
    text = " ".join(text.split())

    if not text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]

def chunk_section(section, chunk_size=1200, overlap=200):
    """
    Seperate section into chunk have the size smaller than 1200
    """
    sentences = split_into_sentences(section["text"])

    # Bước 1: Bảo đảm mỗi đơn vị không vượt chunk_size.
    units = []

    for sentence in sentences:
        if len(sentence) <= chunk_size:
            units.append(sentence)
            continue

        # Nếu câu quá dài, chia theo từ.
        words = sentence.split()
        piece = ""

        for word in words:
            candidate = f"{piece} {word}".strip()

            if len(candidate) > chunk_size:
                if piece:
                    units.append(piece)

                # Trường hợp một từ riêng lẻ quá dài.
                while len(word) > chunk_size:
                    units.append(word[:chunk_size])
                    word = word[chunk_size:]

                piece = word
            else:
                piece = candidate

        if piece:
            units.append(piece)

    # Bước 2: Ghép các đơn vị thành chunk.
    chunks = []
    start = 0

    while start < len(units):
        end = start
        current_length = 0

        while end < len(units):
            extra = len(units[end])
            if end > start:
                extra += 1  # Dấu cách giữa hai đơn vị.

            if current_length + extra > chunk_size:
                break

            current_length += extra
            end += 1

        # Bảo đảm thuật toán luôn tiến về phía trước.
        if end == start:
            end = start + 1

        chunk_text = " ".join(units[start:end]).strip()

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "section": section["section"]
            }
        })

        if end >= len(units):
            break

        # Bước 3: Chọn vị trí bắt đầu tiếp theo để tạo overlap.
        next_start = end
        overlap_length = 0

        while next_start > start:
            candidate_length = len(units[next_start - 1])
            if next_start < end:
                candidate_length += 1

            if overlap_length + candidate_length > overlap:
                break

            overlap_length += candidate_length
            next_start -= 1

        # Luôn tiến ít nhất một đơn vị để tránh vòng lặp vô hạn.
        start = max(start + 1, next_start)

    return chunks

def chunk_markdown(text, source, chunk_size=1200, overlap=200):
    """
    Create chunk from md file and save source + section
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    sections = split_into_sections(text)
    chunks = []

    for section in sections:
        if section["section"].strip().upper() == "CONTENTS":
            continue

        section_chunks = chunk_section(
            section,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in section_chunks:
            chunk["metadata"]["source"] = source
            chunks.append(chunk)

    return chunks

