# Release Discipline & Change Control

Every change to Voice Soundboard is classified before merging.
This prevents accidental breakage and keeps users informed.

---

## Change Classification

| Type | Version Bump | Criteria | Example |
|------|-------------|----------|---------|
| **Patch** | 1.2.x | Bug fix, typo, docs-only, test-only | Fix off-by-one in speed clamping |
| **Minor** | 1.x.0 | New feature, new optional parameter, deprecation | Add `normalize=False` option |
| **Breaking** | x.0.0 | Removes/renames stable API, changes return type, changes default behavior | Rename `speak()` parameter |

### What counts as breaking (requires major version bump)

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full list. Summary:

- Removing or renaming anything in `__init__.py`
- Changing `SpeechResult` field names or types
- Changing `VoiceEngine.speak()` or `speak_raw()` signatures
- Changing the parameter priority order
- Changing the default voice or default speed
- Changing the default output format (WAV)

### What does NOT count as breaking

- Adding a new optional parameter to `speak()`
- Adding a new emotion, preset, or voice
- Adding a new subpackage
- Changing experimental module APIs (codecs, conversion, llm, vocology, studio)
- Performance improvements
- Internal refactoring

---

## PR Checklist

Every pull request must answer these questions before merging:

```markdown
## PR Checklist

- [ ] **Change type:** patch / minor / breaking
- [ ] **API impact:** Does this change any symbol in `__init__.py`? If yes, is it additive-only?
- [ ] **Flags required?** Does this introduce new behavior that should be opt-in?
- [ ] **Tests:** Are there tests for the change? Do smoke tests still pass?
- [ ] **CHANGELOG:** Is there a changelog entry under `[Unreleased]`?
- [ ] **Docs:** Are docs updated if this changes user-facing behavior?
- [ ] **Glossary:** Does this introduce new terminology? If yes, is it in GLOSSARY.md?
```

---

## CHANGELOG Format

We use [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Every PR adds an entry under `## [Unreleased]` in one of these sections:

- **Added** - New features
- **Changed** - Changes to existing behavior
- **Deprecated** - Features that will be removed
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Vulnerability fixes

When releasing, `[Unreleased]` becomes `[X.Y.Z] - YYYY-MM-DD`.

---

## Release Process

1. All smoke tests pass (`pytest tests/smoke/ -v`)
2. CHANGELOG `[Unreleased]` section is non-empty
3. Version bumped in `__init__.py` (`__version__` and `API_VERSION` if breaking)
4. Tag the release: `git tag vX.Y.Z`
5. Push tag: `git push origin vX.Y.Z`
6. CI publishes to PyPI

---

## Core Surface Freeze

The core API surface (`__init__.py` exports) is frozen per Phase 3 rules.
Changes to frozen interfaces require:

1. A deprecation warning for 1 minor release
2. Explicit justification in the PR
3. `API_VERSION` bump
4. CHANGELOG entry under "Breaking"
