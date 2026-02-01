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

      perSystem =
        { pkgs, system, ... }:
        let
          treefmtEval = treefmt-nix.lib.evalModule pkgs {
            projectRootFile = "flake.nix";
            programs = {
              nixfmt.enable = true;
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

          pre-commit-check = git-hooks.lib.${system}.run {
            src = ./.;
            hooks = {
              gitleaks = {
                enable = true;
                name = "gitleaks";
                entry = "${pkgs.gitleaks}/bin/gitleaks protect --staged --config .gitleaks.toml";
                language = "system";
                pass_filenames = false;
              };
              treefmt = {
                enable = true;
                package = treefmtEval.config.build.wrapper;
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
        in
        {
          formatter = treefmtEval.config.build.wrapper;

          checks = {
            formatting = treefmtEval.config.build.check ./.;

            gitleaks =
              pkgs.runCommand "check-gitleaks"
                {
                  nativeBuildInputs = [ pkgs.gitleaks ];
                  src = pkgs.lib.fileset.toSource {
                    root = ./.;
                    fileset = pkgs.lib.fileset.gitTracked ./.;
                  };
                }
                ''
                  cd $src
                  gitleaks detect --source . --config .gitleaks.toml --no-git
                  touch $out
                '';

            uv-lock =
              pkgs.runCommand "check-uv-lock"
                {
                  nativeBuildInputs = [
                    pkgs.uv
                    pkgs.cacert
                  ];
                  src = pkgs.lib.fileset.toSource {
                    root = ./.;
                    fileset = pkgs.lib.fileset.gitTracked ./.;
                  };
                  SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
                }
                ''
                  cd $src
                  export HOME=$(mktemp -d)
                  uv lock --check
                  touch $out
                '';

            ty =
              pkgs.runCommand "check-ty"
                {
                  nativeBuildInputs = [
                    pkgs.ty
                    pkgs.uv
                    pkgs.python313
                    pkgs.cacert
                  ];
                  src = pkgs.lib.fileset.toSource {
                    root = ./.;
                    fileset = pkgs.lib.fileset.gitTracked ./.;
                  };
                  SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
                }
                ''
                  cp -r $src/. ./workdir
                  chmod -R u+w ./workdir
                  cd ./workdir

                  export HOME=$(mktemp -d)
                  export UV_LINK_MODE=copy

                  uv sync --all-extras --locked --python ${pkgs.python313}/bin/python3.13
                  uv run ty check stackone_ai
                  touch $out
                '';

            pytest =
              pkgs.runCommand "check-pytest"
                {
                  nativeBuildInputs = [
                    pkgs.uv
                    pkgs.python313
                    pkgs.bun
                    pkgs.pnpm_10
                    pkgs.typescript-go
                    pkgs.git
                    pkgs.cacert
                  ];
                  src = pkgs.lib.fileset.toSource {
                    root = ./.;
                    fileset = pkgs.lib.fileset.gitTracked ./.;
                  };
                  SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
                }
                ''
                  cp -r $src/. ./workdir
                  chmod -R u+w ./workdir
                  cd ./workdir

                  export HOME=$(mktemp -d)
                  export UV_LINK_MODE=copy

                  # Initialize git submodules
                  git init
                  git submodule update --init --recursive || true

                  # Install dependencies and run tests
                  uv sync --all-extras --locked --python ${pkgs.python313}/bin/python3.13
                  uv run pytest
                  touch $out
                '';
          };

          devShells.default = pkgs.mkShellNoCC {
            buildInputs = with pkgs; [
              uv
              ty
              just
              nixfmt

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
                uv sync --all-extras --locked
              fi

              # Install git hooks
              ${pre-commit-check.shellHook}
            '';
          };
        };
    };
}
