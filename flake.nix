{
  description = "StackOne AI Python SDK development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    git-hooks.url = "github:cachix/git-hooks.nix";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs =
    inputs@{
      flake-parts,
      git-hooks,
      treefmt-nix,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      imports = [
        git-hooks.flakeModule
        treefmt-nix.flakeModule
      ];

      perSystem =
        {
          config,
          pkgs,
          ...
        }:
        {
          # Treefmt configuration for formatting
          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              nixfmt.enable = true;
              nixfmt.package = pkgs.nixfmt-rfc-style;
              ruff-check.enable = true;
              ruff-format.enable = true;
              prettier = {
                enable = true;
                includes = [
                  "*.md"
                  "*.yml"
                  "*.yaml"
                  "*.json"
                ];
                excludes = [
                  "CHANGELOG.md"
                ];
              };
              typos.enable = true;
            };
          };

          # Git hooks configuration
          pre-commit = {
            check.enable = false; # Skip check in flake (ty needs Python env)
            settings.hooks = {
              treefmt = {
                enable = true;
                package = config.treefmt.build.wrapper;
              };
              ty = {
                enable = true;
                name = "ty";
                entry = "${pkgs.uv}/bin/uv run ty check";
                files = "^stackone_ai/";
                language = "system";
                types = [ "python" ];
              };
            };
          };

          devShells.default = pkgs.mkShell {
            buildInputs = with pkgs; [
              uv
              just
              nixfmt-rfc-style
              typos
              typos-lsp
            ];

            shellHook = ''
              echo "StackOne AI Python SDK development environment"

              # Install dependencies only if .venv is missing or uv.lock is newer
              if [ ! -d .venv ] || [ uv.lock -nt .venv ]; then
                echo "📦 Installing dependencies..."
                uv sync --all-extras
              fi

              # Install git hooks
              ${config.pre-commit.installationScript}
            '';
          };
        };
    };
}
