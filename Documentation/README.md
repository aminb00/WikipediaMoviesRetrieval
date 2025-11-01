# Documentation - LaTeX Report

## How to Compile

### Option 1: Online (Overleaf)
1. Upload all files in this folder to Overleaf
2. Compile with pdfLaTeX
3. Note: Logo image is optional, remove the line if not available

### Option 2: Local (TeXLive/MiKTeX)
```bash
pdflatex main.tex
pdflatex main.tex  # Run twice for TOC
```

## Customization

### Update Group Members
Edit lines 56-61 in `main.tex`:
```latex
\begin{tabular}{ll}
    Your Name 1 & Your ID 1 \\
    Your Name 2 & Your ID 2 \\
    Your Name 3 & Your ID 3 \\
    Your Name 4 & Your ID 4 \\
\end{tabular}
```

### Add University Logo (Optional)
1. Download University of Antwerp logo
2. Save as `uantwerp_logo.png` in this folder
3. Or comment out line 52 in main.tex if logo not available:
```latex
% \includegraphics[width=0.3\textwidth]{uantwerp_logo.png} \\[1cm]
```

### Change Colors
Edit line 19 in `main.tex`:
```latex
\definecolor{uablue}{RGB}{0,61,165}  % Change RGB values
```

## Features

✅ Professional title page with university branding
✅ Table of contents with hyperlinks
✅ Proper section numbering
✅ Header/footer with assignment info
✅ Code syntax highlighting
✅ University colors (blue theme)
✅ Proper citations and references
✅ A4 paper format with standard margins

## Document Structure

The report follows a clean, linear structure:

1. **Title Page**: University branding, project title, group members
2. **Table of Contents**: Auto-generated with hyperlinks
3. **Introduction**: System overview and dataset description
4. **Tokenizer**: Text preprocessing (to be completed)
5. **Indexer**: Three variants (memory SPIMI, disk-based, updatable)
   - Memory Index – SPIMI Approach
   - Disk Index – Term-per-File and Lazy Loading
   - Updatable Index – Auxiliary Index and Merge
6. **Query Processing**: Ranking algorithms (to be completed)
   - Vector Space Model (VSM)
   - BM25 Ranking
   - Language Models
7. **Conclusion**: Summary and results (to be completed)
8. **References**: Properly formatted bibliography

## Notes

- Sections marked "[To be completed]" are placeholders for future content
- Indexer section contains complete implementation details
- No references to assignment point system
- Professional academic writing style throughout
