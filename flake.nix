{
  description = "OpenVAD: training, inference engine and thesis";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # The docs shell builds tudcd-scr from the submodule.
    self.submodules = true;
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      git-hooks,
    }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

      # One attrset so other shells can reuse it (mkShell inputsFrom drops env vars).
      shellEnv =
        pkgs:
        {
          UV_PYTHON = "${pkgs.python312}/bin/python3";
          UV_PYTHON_DOWNLOADS = "never";
          UV_PROJECT_ENVIRONMENT = ".venv";
        }
        // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          LD_LIBRARY_PATH =
            pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ]
            + ":/run/opengl-driver/lib";
        };

      # TeX Live set for the thesis; extend only when LuaLaTeX reports "file not found".
      texFor =
        pkgs:
        pkgs.texliveBasic.withPackages (
          ps: with ps; [
            koma-script
            fontspec
            luaotfload
            babel-english
            biblatex
            biber
            acro
            siunitx
            pdfpages
            pgf
            l3build
            latexmk
            chktex
            latexindent
            # biblatex warns without csquotes when babel is loaded.
            csquotes
            # Required by tudcd-scr and the packages it loads.
            fontaxes
            newtx
            noto
            notomath
            pdflscape
            qrcode
            ragged2e
            translations
            txfonts
            xkeyval
            xurl
          ]
        );

      # tudcd-scr ships only .dtx sources; unpack them into a TEXMFHOME tree.
      tudcdFor =
        pkgs:
        pkgs.stdenvNoCC.mkDerivation {
          pname = "tudcd-scr";
          version = "0.7.1";
          src = ./docs/tudcd-scr;
          nativeBuildInputs = [ (pkgs.texliveBasic.withPackages (ps: [ ps.l3build ])) ];
          buildPhase = "HOME=$TMPDIR l3build unpack";
          installPhase = ''
            mkdir -p $out/tex/latex/tudcd-scr
            cp build/unpacked/*.cls build/unpacked/*.sty $out/tex/latex/tudcd-scr/
            cp -r source/logo $out/tex/latex/tudcd-scr/
          '';
        };

      # withUvLock is off for checks: the Nix sandbox has no network.
      hooksFor =
        system: withUvLock:
        let
          pkgs = pkgsFor system;
          rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        in
        git-hooks.lib.${system}.run {
          src = ./.;
          # Third-party submodule (pinned upstream), must not be reformatted.
          excludes = [ "^docs/tudcd-scr/" ];
          # The stock pre-commit test deps (dotnet, swift) have no darwin cache and build from source.
          package = pkgs.pre-commit.overridePythonAttrs {
            doCheck = false;
            dontUsePytestCheck = true;
            nativeCheckInputs = [ ];
            preCheck = "";
          };
          hooks = {
            ruff.enable = true;
            ruff-format.enable = true;
            rustfmt = {
              enable = true;
              packageOverrides = {
                cargo = rust;
                rustfmt = rust;
              };
            };
            nixfmt.enable = true;
            typos.enable = true;
            end-of-file-fixer.enable = true;
            trim-trailing-whitespace.enable = true;
            check-added-large-files = {
              enable = true;
              args = [ "--maxkb=500" ];
            };
          }
          // pkgs.lib.optionalAttrs withUvLock {
            uv-lock = {
              enable = true;
              name = "uv-lock";
              entry = "${pkgs.uv}/bin/uv lock --check";
              files = "(^|/)(pyproject\\.toml|uv\\.lock)$";
              pass_filenames = false;
            };
          };
        };
    in
    {
      formatter = forAllSystems (system: (pkgsFor system).nixfmt);

      checks = forAllSystems (system: {
        pre-commit = hooksFor system false;
      });

      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
          default = pkgs.mkShell (
            shellEnv pkgs
            // {
              packages = [
                rust
                pkgs.maturin
                pkgs.uv
                pkgs.ffmpeg-full
                pkgs.python312
                pkgs.just
                pkgs.typos
                pkgs.nixfmt
              ];
              inherit (hooksFor system true) shellHook;
            }
          );
        in
        {
          inherit default;
          # Tools for `just build-mfa` (tools/README.md): enter with `nix develop .#align`.
          align = pkgs.mkShell (
            shellEnv pkgs
            // {
              inputsFrom = [ default ];
              packages = [
                pkgs.cmake
                pkgs.curl
                (pkgs.postgresql.withPackages (ps: [ ps.pgvector ]))
              ];
            }
            // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
              OPENBLAS_ROOT = pkgs.symlinkJoin {
                name = "openblas-root";
                paths = [
                  pkgs.openblas
                  pkgs.openblas.dev
                ];
              };
            }
          );
          docs = pkgs.mkShell (
            shellEnv pkgs
            // {
              inputsFrom = [ default ];
              packages = [ (texFor pkgs) ];
              TEXMFHOME = tudcdFor pkgs;
            }
          );
        }
      );
    };
}
