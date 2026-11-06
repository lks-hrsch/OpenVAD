# docs

- Documents compile with LuaLaTeX (`$pdf_mode = 4` in `latexmkrc`).
- Read `tudcd-scr/doc/handbook.tex`, `tudcd-scr/doc/demo-*.tex` and `tudcd-scr/source/*.dtx` before you set any tudcd class option or `\TUDCDsetup` key; use only what they document.
- Write one sentence per line in `.tex` files.
- Never edit files under `generated/`; change `openvad_train.export_docs` and run `just docs-generated`.
