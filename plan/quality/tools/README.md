# Audit tooling

Read-only scripts that produced the numbers in `plan/quality/`. Nothing here
changes a repository. Python 3 standard library only.

```bash
ROOT=/path/to/repo
python3 fnscan.py  $ROOT fns.json              # every fn: length, nesting, branches, params
python3 longfns.py $ROOT fns.json 90          # markdown table of production fns >= 90 lines
python3 dup.py     $ROOT 8                     # share of production lines in duplicated 8-line windows
./lints.sh $ROOT out.clippy.json --workspace --lib --bins
python3 agg.py out.clippy.json                 # clippy hits by lint and file
```

Caveats, stated plainly:
- `fnscan.py` is a brace matcher, not a parser. It strips comments and string
  literals first; macro-generated functions and `fn` inside macro bodies can be
  mis-measured. Treat lengths as accurate to a few lines.
- "Branch points" counts `if`, `match`, `for`, `while`, `loop`, `?;`, `&&`, `||`.
  It is a proxy for cyclomatic complexity, not the real metric.
- `dup.py` finds repeated text after normalization, not semantic clones. A hit
  is a lead to read, not proof the code should merge.
