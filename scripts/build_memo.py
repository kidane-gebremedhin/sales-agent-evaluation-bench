import markdown
from weasyprint import HTML, CSS
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_MD = REPO / "memo.md"
OUT_PDF = REPO / "memo.pdf"

PRINT_CSS = """
@page {
  size: Letter;
  margin: 22mm 18mm 22mm 18mm;
  @bottom-right {
    content: "Tenacious Executive Memo — page " counter(page) " of " counter(pages);
    font-family: 'Helvetica', sans-serif;
    font-size: 8pt;
    color: #777;
  }
  @bottom-left {
    content: "Kidane · W11 · 2026-05-02";
    font-family: 'Helvetica', sans-serif;
    font-size: 8pt;
    color: #777;
  }
}
body {
  font-family: 'Helvetica', 'Arial', sans-serif;
  font-size: 11pt;
  line-height: 1.5;
  color: #1a1a1a;
}
h1 { font-size: 20pt; margin-top: 0; padding-top: 0.3em; border-bottom: 2px solid #1f4068; padding-bottom: 0.2em; color: #1f4068;}
h1:first-of-type { margin-top: 0; }
h2 { font-size: 14pt; margin-top: 1.2em; color: #1f4068; border-bottom: 1px solid #c8d3e0; padding-bottom: 0.15em; }
h3 { font-size: 11pt; margin-top: 1em; color: #233863; }
p, ul, ol { margin-top: 0.8em; margin-bottom: 0.8em; }
ul, ol { padding-left: 1.6em; }
li { margin: 0.4em 0; }
strong { color: #1a1a1a; }
hr { border: 0; border-top: 1px solid #c8d3e0; margin: 1.5em 0; }
"""

def _strip_frontmatter(md: str) -> str:
    if md.startswith("---\n"):
        end = md.find("\n---\n", 4)
        if end != -1:
            return md[end + 5 :]
    return md

def main():
    md_text = OUT_MD.read_text()
    html_md = _strip_frontmatter(md_text)
    html = markdown.markdown(html_md, extensions=["sane_lists", "attr_list"])
    css = CSS(string=PRINT_CSS)
    HTML(string=html, base_url=str(REPO)).write_pdf(OUT_PDF, stylesheets=[css])
    print(f"Generated {OUT_PDF}")

if __name__ == "__main__":
    main()
