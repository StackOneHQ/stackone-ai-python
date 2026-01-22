# Nix Workflow

This rule provides guidance on Nix usage in the StackOne AI Python SDK.

## Development Environment

The project uses `flake.nix` with flake-parts to define the development environment. Enter it with `nix develop`.

### Adding Development Tools

To add a new tool to the development environment, add it to `buildInputs` in `flake.nix`:

```nix
devShells.default = pkgs.mkShellNoCC {
  buildInputs = with pkgs; [
    uv
    ty
    just
    nixfmt-rfc-style

    # your new tool here
    new-tool
  ];
};
```

### Treefmt and Git Hooks

The flake includes:

- **treefmt-nix**: Unified formatting (nixfmt, ruff, oxfmt)
- **git-hooks.nix**: Pre-commit hooks (gitleaks, treefmt, ty)

These are automatically installed when entering the dev shell.

## CI Workflow

CI uses `nix profile install` via the `.github/actions/setup-nix/action.yaml` composite action.

### Adding Tools to CI Jobs

Specify tools in the `tools` input of the setup-nix action:

```yaml
- name: Setup Nix
  uses: ./.github/actions/setup-nix
  with:
    tools: uv ty just bun pnpm_10
```

The action installs packages using:

```bash
nix profile install --inputs-from . nixpkgs#tool1 nixpkgs#tool2
```

### CI Tool Configuration

- **Default tools**: `uv ty just` (defined in action.yaml)
- **Skip uv sync**: Set `skip-uv-sync: 'true'` for jobs that don't need Python dependencies

### Example: Adding a New Tool to CI Job

```yaml
ci:
  runs-on: ubuntu-latest
  steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      with:
        submodules: true
    - name: Setup Nix
      uses: ./.github/actions/setup-nix
      with:
        tools: uv ty just new-tool
    - name: Run Lint
      run: just lint
```

## Notes

- The project uses flake-parts for modular flake configuration
- Git submodules are initialised automatically in dev shell and CI
- MCP mock server dependencies (pnpm) are installed for testing
