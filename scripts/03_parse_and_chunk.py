"""Parse LaTeX sources into clean text chunks.

For each downloaded .src file:
1. Detect format (tar.gz / gzipped single / raw)
2. Extract all .tex files, find the main one (has \\documentclass)
3. Strip bibliography, figures, comments
4. Convert LaTeX -> text via pylatexenc, keeping math verbatim
5. Sliding-window chunk by word count

Output: data/chunks/chunks.jsonl (one chunk per line)
"""
import gzip
import io
import json
import re
import tarfile
from pathlib import Path

from pylatexenc.latex2text import LatexNodes2Text
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "data" / "metadata" / "papers.jsonl"
SOURCES = ROOT / "data" / "sources"
CHUNKS_OUT = ROOT / "data" / "chunks" / "chunks.jsonl"
CHUNKS_OUT.parent.mkdir(parents=True, exist_ok=True)

CHUNK_WORDS = 450      # ~600 tokens
OVERLAP_WORDS = 60


def safe_filename(aid):
    return aid.replace("/", "_")


def arxiv_id_no_version(aid):
    last = aid.rsplit("/", 1)[-1]
    if "v" in last:
        base = "v".join(last.split("v")[:-1])
        prefix = aid.rsplit("/", 1)[0] + "/" if "/" in aid else ""
        return prefix + base
    return aid


def _extract_tar(fileobj):
    """Pull all .tex files from a tar archive (any compression)."""
    out = []
    for mode in ("r:gz", "r:bz2", "r:", "r"):
        try:
            with tarfile.open(fileobj=fileobj, mode=mode) as tar:
                for m in tar.getmembers():
                    if m.isfile() and m.name.lower().endswith(".tex"):
                        f = tar.extractfile(m)
                        if f:
                            out.append((m.name, f.read().decode("utf-8", errors="replace")))
            return out
        except Exception:
            try:
                fileobj.seek(0)
            except Exception:
                pass
    return []


def _looks_like_latex(text):
    """True if the text contains typical LaTeX commands."""
    return any(tok in text for tok in (
        "\\documentclass", "\\begin{document}", "\\input", "\\section",
        "\\overfullrule", "\\def", "\\newcommand",
    ))


def detect_and_extract(src_bytes):
    """Return list of (filename, content_str) for all .tex files."""
    # Try tar.gz directly
    result = _extract_tar(io.BytesIO(src_bytes))
    if result:
        return result

    # Try gzip decompression first, then re-try as tar or plain text
    try:
        decompressed = gzip.decompress(src_bytes)
        # Maybe the gzip wraps a tar
        result = _extract_tar(io.BytesIO(decompressed))
        if result:
            return result
        # Or it's a single LaTeX file
        content = decompressed.decode("utf-8", errors="replace")
        if _looks_like_latex(content):
            return [("main.tex", content)]
    except Exception:
        pass

    # Try raw text (uncompressed .tex)
    try:
        content = src_bytes.decode("utf-8", errors="replace")
        if _looks_like_latex(content):
            return [("main.tex", content)]
    except Exception:
        pass

    return []


def find_main(tex_files):
    """Pick the file with \\documentclass; fall back to shortest filename."""
    with_doc = [(n, c) for n, c in tex_files if "\\documentclass" in c]
    pool = with_doc or tex_files
    if not pool:
        return None
    pool.sort(key=lambda x: len(x[0]))
    return pool[0]


def strip_cruft(latex):
    # Keep only the body: between \begin{document} and \end{document}.
    # This is critical — preambles contain hundreds of macro definitions
    # that pylatexenc renders as garbage text.
    m_begin = re.search(r"\\begin\{document\}", latex)
    m_end = re.search(r"\\end\{document\}", latex)
    if m_begin:
        start = m_begin.end()
        end = m_end.start() if m_end else len(latex)
        latex = latex[start:end]
    # Strip comments (% to EOL, but not escaped \%)
    latex = re.sub(r"(?<!\\)%.*", "", latex)
    # Strip bibliography
    latex = re.sub(r"\\bibliography\{[^}]*\}", "", latex)
    latex = re.sub(r"\\bibliographystyle\{[^}]*\}", "", latex)
    latex = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
                   "", latex, flags=re.DOTALL)
    # Replace figure / table environments with placeholders
    latex = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}",
                   "[FIGURE]", latex, flags=re.DOTALL)
    latex = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}",
                   "[TABLE]", latex, flags=re.DOTALL)
    # Drop \input{} and \include{} — we already gathered all .tex files;
    # following these would just duplicate or fail.
    latex = re.sub(r"\\input\{[^}]*\}", "", latex)
    latex = re.sub(r"\\include\{[^}]*\}", "", latex)
    return latex


_converter = LatexNodes2Text(
    math_mode="verbatim",      # keep $...$, \[...\], etc. verbatim
    keep_comments=False,
    strict_latex_spaces=False,
)


def latex_to_text(latex):
    try:
        text = _converter.latex_to_text(strip_cruft(latex))
    except Exception:
        return ""
    # Clean up pylatexenc's placeholder markers
    text = re.sub(r"\[NO\s+\\[a-z]+\s+GIVEN\]", "", text)
    text = re.sub(r"<cit\.>", "[cite]", text)
    # Collapse runs of whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, words_per_chunk=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def main():
    papers = [json.loads(l) for l in METADATA.open()]
    parsed = no_src = no_tex = empty = 0
    total_chunks = 0

    with CHUNKS_OUT.open("w") as out:
        for paper in tqdm(papers, desc="Parsing"):
            aid = arxiv_id_no_version(paper["arxiv_id"])
            src_path = SOURCES / f"{safe_filename(aid)}.src"

            if not src_path.exists():
                no_src += 1
                continue

            tex_files = detect_and_extract(src_path.read_bytes())
            if not tex_files:
                no_tex += 1
                continue

            main_pair = find_main(tex_files)
            if not main_pair:
                no_tex += 1
                continue

            text = latex_to_text(main_pair[1])
            if len(text.split()) < 50:
                empty += 1
                continue

            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                rec = {
                    "chunk_id": f"{safe_filename(aid)}_{idx:03d}",
                    "arxiv_id": aid,
                    "title": paper["title"],
                    "year": paper["published"][:4],
                    "primary_category": paper["primary_category"],
                    "chunk_idx": idx,
                    "n_chunks": len(chunks),
                    "text": chunk,
                }
                out.write(json.dumps(rec) + "\n")
                total_chunks += 1
            parsed += 1

    print(f"\nParsed papers:   {parsed}")
    print(f"Total chunks:    {total_chunks}")
    print(f"No source file:  {no_src}")
    print(f"No .tex inside:  {no_tex}")
    print(f"Empty/tiny:      {empty}")
    print(f"Output: {CHUNKS_OUT}")


if __name__ == "__main__":
    main()
