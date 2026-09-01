# Setup

## 1. Drop it in

Copy the scaffold over your existing repo root. It adds files; it doesn't move
your notes.

```
AGENT.md            the contract — portable across harnesses
CLAUDE.md           one line: @AGENT.md, plus Claude Code specifics
.claude/
  settings.json     permissions (see §3 — this is the important part)
  commands/         /s1 /s2 /s3 /s4 /cold-open /unblock /self-test /lint
                    /week-plan /wrap
  agents/           note-reviewer, librarian
tools/
  vault_lint.py     structural linter
  new_note.py       frontmatter + skeleton generator
prompts/
  tutor_system_prompt.md    for Codex, Cursor, aider, or a raw API loop
vault/  mini/  tests/  scratch/
```

Then move your existing files in: notes under `vault/Q1/`, evergreen concept
notes under `vault/topics/`, `.py` implementations under `mini/`.

Claude Code reads `CLAUDE.md`, not `AGENT.md` — that's why both exist. `CLAUDE.md`
is a one-line `@AGENT.md` import, so there's a single source of truth and no
drift. Other harnesses that follow the agents.md convention read `AGENT.md`
directly.

## 2. First run — the Week 14 cleanup

```bash
python3 tools/vault_lint.py vault/ --quiet-info
```

Against your current files this reports roughly 30 errors and 80 warnings. In
order of what to do about them:

**`E1` no extension** — `w13s3-reading` and `Q1_revised_plan_weeks_7-14` have no
`.md`, so Obsidian isn't indexing them as notes. Safe to auto-fix.

**`E2`/`E3` frontmatter** — about twenty notes have no frontmatter block, and two
have partial blocks. `--fix` adds them, deriving `created` from file mtime and
generating a proper UUIDv7. Also safe.

```bash
git commit -am "pre-lint checkpoint"
python3 tools/vault_lint.py vault/ --fix
```

**`W2` duplicate slot** — `Week11_S1_ImplicitBias_Math.md` and
`w11s1-implicit-bias-math.md` both claim Week 11 S1. This is your naming-drift
problem and it is a merge decision, not a rename. The linter refuses to
auto-rename either one (it prints `HOLD`), because renaming the legacy file
would produce `Week11_S1_Implicit_Bias_Math.md` — a near-identical lookalike
sitting next to the original. Open both, decide which survives, delete the other.

**`W1` filename drift** — the `w7s1-momentum-math` family. After the duplicates
are resolved:

```bash
python3 tools/vault_lint.py vault/ --rename            # dry run, read it
python3 tools/vault_lint.py vault/ --rename --apply    # rewrites wiki-links too
```

Do this with a clean git tree. It rewrites every `[[link]]` pointing at a renamed
file, which is a lot of edits at once.

**`E4` broken session links** — seven of these, including `[[Week12_S1_Conv_Math]]`
and `[[Week4_...]]`, which is a literal placeholder that never got filled in.
Human decisions, one at a time.

**`W4` unresolved concept links** — `[[Hessian]]`, `[[eigenvalue]]`,
`[[Condition Number]]` and friends. These are intentional Zettelkasten stubs. The
linter flags them as warnings rather than errors so you can decide when to
promote them:

```bash
python3 tools/new_note.py --kind topic --title "Condition Number"
```

## 3. The permissions are the design

`.claude/settings.json` denies `Write(vault/**)` and `Write(mini/**)`.

This is deliberate, and it's the piece that makes the whole thing work. A rule in
a markdown file that says "don't write the derivation" is a suggestion the model
can talk itself out of at 11pm when you're tired and it's trying to be helpful.
A deny rule is enforced by the harness before the model gets a vote. Drafts go to
`scratch/`, and moving something from `scratch/` into `vault/` is a deliberate act
you perform yourself.

Two honest caveats. Deny rules cover the built-in file tools and the file
commands Claude Code recognizes in Bash (`cat`, `sed`, and similar) — they don't
stop a Python script that opens a file itself. And `--dangerously-skip-permissions`
bypasses the whole system by design. Neither matters much here, since the threat
model is your own convenience rather than an adversary.

## 4. A week, end to end

```
/lint                                 # start clean
/s1 15 "2D positional encoding"       # cold open → derivation → label check
/s2 15 "sinusoidal PE, NumPy"         # predictions gate → implement → compare
/s3 15 "Dosovitskiy et al., ViT"      # beliefs → claims/demos/conclusions
/s4 15 "PE Explainer"                 # draft → numbered issues → revise
/wrap                                 # calibration, hooks, lint, commit
```

`/unblock` when stuck. `/self-test 4-14` closed-book, ideally cold, ideally on a
day you don't feel like it.

## 5. Q2 and the NumPy/PyTorch decision

If Q2 moves to PyTorch, one rule in `AGENT.md` needs revisiting: "never write the
mathematical core." In NumPy that boundary is obvious, because the core *is* the
math. In PyTorch, `nn.MultiheadAttention` is one line and the derivation is
invisible. The boundary that preserves the pedagogy is roughly: the agent may
write anything that would be boilerplate in a paper's appendix — dataloaders,
training loops, logging — and nothing that would appear in the paper's method
section.

Worth writing that rule down explicitly before the first PyTorch session rather
than after, since the failure is silent.

## 6. Other harnesses

`prompts/tutor_system_prompt.md` is the same contract compressed into a system
prompt. Codex CLI and Cursor read `AGENT.md`/`AGENTS.md` directly; aider takes
`--read AGENT.md`; a raw API loop takes the prompt file as its system message.

The slash commands don't port — but their bodies are just prompts, so pasting
`.claude/commands/s1.md` into a conversation works fine anywhere.
