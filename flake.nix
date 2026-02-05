{
  description = "StackOne AI Python SDK development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    git-hooks.url = "github:cachix/git-hooks.nix";
    treefmt-nix.url = "github:numtide/treefmt-nix";

    # Agent skills management
    agent-skills.url = "github:Kyure-A/agent-skills-nix";
    agent-skills.inputs.nixpkgs.follows = "nixpkgs";

    # StackOne skills repository (non-flake)
    stackone-skills.url = "github:StackOneHQ/skills";
    stackone-skills.flake = false;
  };

  outputs =
    inputs@{
      flake-parts,
      git-hooks,
      treefmt-nix,
      agent-skills,
      stackone-skills,
      ...
    }:
    let
      # Agent skills configuration (outside flake-parts for access to inputs)
      agentLib = agent-skills.lib.agent-skills;
      sources = {
        stackone = {
          path = stackone-skills;
          subdir = ".";
        };
      };
      catalog = agentLib.discoverCatalog sources;
      allowlist = agentLib.allowlistFor {
        inherit catalog sources;
        enable = [
          "just-commands"
          "release-please"
        ];
      };
      selection = agentLib.selectSkills {
        inherit catalog allowlist sources;
        skills = { };
      };
    in
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
              options = [ "--no-error-on-unmatched-pattern" ];
              includes = [ "*" ];
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

          # Agent skills bundle and targets
          bundle = agentLib.mkBundle { inherit pkgs selection; };
          # Use symlink-tree instead of copy-tree for skills
          localTargets = inputs.nixpkgs.lib.mapAttrs (
            _: t: t // { structure = "symlink-tree"; }
          ) agentLib.defaultLocalTargets;
        in
        {
          formatter = treefmtEval.config.build.wrapper;

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
                echo "Initializing git submodules..."
                git submodule update --init --recursive
              fi

              # Install Python dependencies only if .venv is missing or uv.lock is newer
              if [ ! -d .venv ] || [ uv.lock -nt .venv ]; then
                echo "Installing Python dependencies..."
                uv sync --all-extras --locked
              fi

              # Install git hooks
              ${pre-commit-check.shellHook}
            ''
            + agentLib.mkShellHook {
              inherit pkgs bundle;
              targets = localTargets;
            };
          };
        };
    };
}
