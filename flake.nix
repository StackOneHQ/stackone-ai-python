{
  description = "StackOne AI Python SDK development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      perSystem =
        { pkgs, ... }:
        {
          devShells.default = pkgs.mkShell {
            buildInputs = with pkgs; [
              uv
              nixfmt-rfc-style
            ];

            shellHook = ''
              echo "StackOne AI Python SDK development environment"

              # Install dependencies only if .venv is missing or uv.lock is newer
              if [ ! -d .venv ] || [ uv.lock -nt .venv ]; then
                echo "📦 Installing dependencies..."
                uv sync --all-extras
              fi
            '';
          };
        };
    };
}
