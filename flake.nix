{
  description = "StackOne AI Python SDK development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    git-hooks.url = "github:cachix/git-hooks.nix";
    treefmt-nix.url = "github:numtide/treefmt-nix";

    # uv2nix inputs
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{
      flake-parts,
      git-hooks,
      treefmt-nix,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      # Load uv2nix workspace
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      # Create overlay from uv.lock
      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      # Editable overlay for development
      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };
    in
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
          system,
          ...
        }:
        let
          # Supported Python versions
          pythonVersions = {
            python311 = pkgs.python311;
            python313 = pkgs.python313;
          };

          # Override for packages that need additional build dependencies
          buildSystemOverrides = final: prev: {
            pypika = prev.pypika.overrideAttrs (old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                final.setuptools
              ];
            });
            # stackone-ai needs editables for editable install
            stackone-ai = prev.stackone-ai.overrideAttrs (old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                final.editables
              ];
            });
          };

          # Helper function to create a Python environment for a given version
          mkPythonEnv =
            python:
            let
              pythonSet =
                (pkgs.callPackage pyproject-nix.build.packages {
                  inherit python;
                }).overrideScope
                  (
                    nixpkgs.lib.composeManyExtensions [
                      pyproject-build-systems.overlays.wheel
                      overlay
                      buildSystemOverrides
                      editableOverlay
                    ]
                  );
            in
            pythonSet.mkVirtualEnv "stackone-ai-${python.pythonVersion}-env" workspace.deps.all;

          # Create virtualenvs for each Python version
          virtualenvs = builtins.mapAttrs (_name: python: mkPythonEnv python) pythonVersions;

          # Default Python version (3.11)
          defaultPython = pythonVersions.python311;
          defaultVirtualenv = virtualenvs.python311;

          # Helper function to create a devShell for a given Python version
          mkDevShell =
            python: virtualenv:
            pkgs.mkShell {
              packages = [
                virtualenv
                pkgs.uv
                pkgs.just
                pkgs.nixfmt-rfc-style
                pkgs.basedpyright

                # security
                pkgs.gitleaks

                # Node.js for MCP mock server
                pkgs.bun
                pkgs.pnpm_10
                pkgs.typescript-go
              ];

              env = {
                # Prevent uv from managing Python - Nix handles it
                UV_NO_SYNC = "1";
                UV_PYTHON = "${python}/bin/python";
                UV_PYTHON_DOWNLOADS = "never";
              };

              shellHook = ''
                echo "StackOne AI Python SDK development environment (Python ${python.pythonVersion})"

                # Set repo root for editable installs
                export REPO_ROOT=$(git rev-parse --show-toplevel)

                # Unset PYTHONPATH to avoid conflicts
                unset PYTHONPATH

                # Initialize git submodules if not already done
                if [ -f .gitmodules ] && [ ! -f vendor/stackone-ai-node/package.json ]; then
                  echo "Initializing git submodules..."
                  git submodule update --init --recursive
                fi

                # Install Node.js dependencies for MCP mock server (used in tests)
                if [ -f vendor/stackone-ai-node/package.json ]; then
                  if [ ! -f vendor/stackone-ai-node/node_modules/.pnpm/lock.yaml ] || \
                     [ vendor/stackone-ai-node/pnpm-lock.yaml -nt vendor/stackone-ai-node/node_modules/.pnpm/lock.yaml ]; then
                    echo "Installing MCP mock server dependencies..."
                    (cd vendor/stackone-ai-node && pnpm install --frozen-lockfile)
                  fi
                fi
              '';
            };
        in
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
                entry = "${defaultVirtualenv}/bin/ty check";
                files = "^stackone_ai/";
                language = "system";
                types = [ "python" ];
              };
            };
          };

          # Development shells for each Python version
          devShells = {
            default = mkDevShell defaultPython defaultVirtualenv;
            python311 = mkDevShell pythonVersions.python311 virtualenvs.python311;
            python313 = mkDevShell pythonVersions.python313 virtualenvs.python313;
          };

          # Package outputs
          packages = {
            default = defaultVirtualenv;
            python311 = virtualenvs.python311;
            python313 = virtualenvs.python313;
          };
        };
    };
}
