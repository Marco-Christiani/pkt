{
  inputs = {
    # Use unstable so we can access newer CUDA (12.8, 13.x) via cudaPackages_12_8 / cudaPackages_13.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-gl-host.url = "github:numtide/nix-gl-host";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      nix-gl-host,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            # allow unfree cuda stack from nixpkgs
            allowUnfree = true;
            # enables cuda hooks and makes cudaPackages produce cuda-enabled builds
            cudaSupport = true;

            # dont set cudaCapabilities unless to reduce build time or target specific GPU archs
            # ref: https://nixos.org/manual/nixpkgs/stable/#cuda-configuring-nixpkgs-for-cuda
            cudaCapabilities = [ "8.6" ];

            # forward compat produces PTX for future GPUs
            # maybe set to false if targeting specific arch and want minimized build outputs
            # cudaForwardCompat = true;
          };
        };

        # Explicitly select cuda toolchain and a compatible host gcc
        # cudaPackages = pkgs.cudaPackages_13;
        cudaPackages = pkgs.cudaPackages_12_8;
        gccHost = pkgs.gcc13;

        nixgl = nix-gl-host.defaultPackage.${system};

        generateClangd = pkgs.writeShellApplication {
          name = "generate-clangd";
          runtimeInputs = with pkgs; [
            python3
            git
            gccHost
            findutils
          ];
          text = ''
            export CUDA_HOME=${cudaPackages.cudatoolkit}
            export CUDA_ARCH=''${CUDA_ARCH:-sm_86}
            exec python3 ${./scripts/generate_clangd.py}
          '';
        };
      in
      {
        devShells.default = pkgs.mkShellNoCC {
          packages = with pkgs; [
            cmake
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cudart
            gccHost
            nixgl
            go-task
          ];
          # LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath packages;

          # Pin host and CUDA compilers for CMake and nvcc
          #  This doesnt appear strictly neccessary rn but it prevents nvcc from
          #    using a different compiler than what we explictly request and have
          #    available in our env (dont know where its getting v14 from when we
          #    dont have this btw).
          CC = "${gccHost}/bin/gcc";
          CXX = "${gccHost}/bin/g++";
          CUDACXX = "${cudaPackages.cudatoolkit}/bin/nvcc";
          # Convenient sets for downstream tooling, again not strictly neccessary for
          #  compiling+running kernels (but I, for one, do use CUDA_HOME)
          # Who knows what is affected by the other sets but yeah idk its solved things before.
          # LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          #   cudaPackages.cuda_cudart
          # ];
          #
          # CPATH = pkgs.lib.makeSearchPath "include" [ ];
          CUDA_HOME = cudaPackages.cudatoolkit;
          # CUDAToolkit_ROOT = cudaPackages.cudatoolkit;
          # CUDA_PATH = cudaPackages.cudatoolkit;

          shellHook = ''
            # this is magic never touch this lol
            export LD_LIBRARY_PATH=$(nixglhost -p):$LD_LIBRARY_PATH
            echo "NIX_ENFORCE_NO_NATIVE=$NIX_ENFORCE_NO_NATIVE"
          '';
        };

        apps.generate-clangd = {
          type = "app";
          program = "${generateClangd}/bin/generate-clangd";
        };

        devShells.profiling = pkgs.mkShell {
          inputsFrom = [ self.devShells.x86_64-linux.default ];

          packages = [
              cudaPackages.nsight_systems # nix-du: ~1.1 / manual diffing: ~2.3GiB / nix-tree: NAR Size: 8.11 KiB | Closure Size: 15.39 MiB | Added Size: 93.75 KiB
              cudaPackages.nsight_compute # nix-du: ~1.3 / manual diffing: 2.5GiB / NAR Size: 4.93 KiB | Closure Size: 15.24 MiB | Added Size: 16.35 KiB
          ];
        };
      }
    );
}
