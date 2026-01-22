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
            };
            settings.formatter.oxfmt = {
              command = "${pkgs.oxfmt}/bin/oxfmt";
              includes = [
                "*.md"
                "*.yml"
                "*.yaml"
                "*.json"
                "*.ts"
                "*.tsx"
                "*.js"
                "*.jsx"
                "*.html"
                "*.css"
              ];
              excludes = [
                "CHANGELOG.md"
              ];
            };
          };

          # Git hooks configuration
          pre-commit = {
            check.enable = false; # Skip check in flake (ty needs Python env)
            settings.hooks = {
              gitleaks = {
                enable = true;
                name = "gitleaks";
                entry = "${pkgs.gitleaks}/bin/gitleaks protect --staged --config .gitleaks.toml";
                language = "system";
                pass_filenames = false;
              };
              treefmt = {
                enable = true;
                package = config.treefmt.build.wrapper;
              };
              ty = {
                enable = true;
                name = "ty";
                entry = "${pkgs.ty}/bin/ty check";
                files = "^stackone_ai/";
                language = "system";
                types = [ "python" ];
              };
            };
          };

          devShells.default = pkgs.mkShellNoCC {
            buildInputs = with pkgs; [
              uv
              ty
              just
              nixfmt-rfc-style

              # security
              gitleaks

              # Node.js for MCP mock server
              bun
              pnpm_10
              typescript-go
            ];

            shellHook = ''
              echo "StackOne AI Python SDK development environment"

              # Initialize git submodules if not already done
              if [ -f .gitmodules ] && [ ! -f vendor/stackone-ai-node/package.json ]; then
                echo "📦 Initializing git submodules..."
                git submodule update --init --recursive
              fi

              # Install Python dependencies only if .venv is missing or uv.lock is newer
              if [ ! -d .venv ] || [ uv.lock -nt .venv ]; then
                echo "📦 Installing Python dependencies..."
                uv sync --all-extras
              fi

              # Install git hooks
              ${config.pre-commit.installationScript}
            '';
          };
        };
    };
}
