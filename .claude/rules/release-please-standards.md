---
description: Standards for using release-please in the repository. (project)
alwaysApply: true
---

# Release Please Standards

Standards for managing releases with release-please in the repository.

## Configuration Files

```
.release-please-config.json    # Release configuration
.release-please-manifest.json  # Version tracking
.github/workflows/release.yml  # Release workflow
```

## Commit Message Format

```bash
# Features (0.1.0 -> 0.2.0)
feat: add new feature
feat!: breaking change feature

# Bug Fixes (0.1.0 -> 0.1.1)
fix: bug fix description

# No Version Change
docs: update readme
chore: update dependencies
test: add new tests
```

## Release Process

1. Push to main branch triggers release-please
2. Release-please creates/updates release PR
3. Merging release PR:
   - Updates CHANGELOG.md
   - Creates GitHub release
   - Publishes to PyPI using UV

## Required Secrets

```
PYPI_API_TOKEN  # For publishing to PyPI
```

## Good Commit Messages

```bash
docs: update installation guide
fix: handle API timeout errors
feat: add new CRM integration
```

## Bad Commit Messages (avoid)

```bash
updated readme
fixed bug in api
added feature
```
