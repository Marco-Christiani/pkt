{
  inputs = {
    # nixpkgs.url = "github:NixOS/nixpkgs/28ace32529a63842e4f8103e4f9b24960cf6c23a";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        # bootstrap import so we can access allowUnfreeCudaPredicate BEFORE config
        # ref: https://nixos.org/manual/nixpkgs/stable/#cuda-configuring-nixpkgs-for-cuda
        bootstrap = import nixpkgs { inherit system; };

        # main nixpkgs import with CUDA enabled
        pkgs = import nixpkgs {
          inherit system;
          config = {
            # use predicate to avoid globally unlocking all unfree packages
            # ref: https://nixos.org/manual/nixpkgs/stable/#sec-allow-unfree
            # allowUnfreePredicate = bootstrap._cuda.lib.allowUnfreeCudaPredicate;
            # allowUnfreePredicate = pkg: builtins.elem (lib.getName pkg) [ "triton" "cuda" "cudnn" ];
            allowUnfreePredicate =
              pkg:
              builtins.elem (nixpkgs.lib.getName pkg) [
                "triton"
                "torch"
              ]
              || bootstrap._cuda.lib.allowUnfreeCudaPredicate pkg;

            # enables CUDA hooks and makes cudaPackages produce CUDA-enabled builds
            cudaSupport = true;

            # dont set cudaCapabilities unless to reduce build time or target specific GPU archs
            # ref: https://nixos.org/manual/nixpkgs/stable/#cuda-configuring-nixpkgs-for-cuda
            cudaCapabilities = [ "8.6" ];

            # forward compat produces PTX for future GPUs
            # maybe set to false if targeting specific arch and want minimized build outputs
            # cudaForwardCompat = true;
          };
        };

        cudaPackages = pkgs.cudaPackages;

        pythonEnv = pkgs.python311.withPackages (
          ps: with ps; [
            numpy
            pip
            torch-bin
          ]
        );

      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            gnumake
            git
            pkg-config
            cudaPackages.cudatoolkit
            pythonEnv
            gcc
            ninja
            meson

            cudaPackages.cuda_cudart # includes libcudart_static.a
            cudaPackages.cutlass
          ];

          LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            cudaPackages.cuda_cudart
          ];

          CPATH = pkgs.lib.makeSearchPath "include" [
            cudaPackages.cutlass
          ];

          NIX_ENFORCE_NO_NATIVE = "0";
          CUDACXX = "${cudaPackages.cudatoolkit}/bin/nvcc";

          shellHook = ''
            python -c "import torch; print('Torch CUDA available:', torch.cuda.is_available())"
            echo "CUDA toolkit version: ${cudaPackages.cudatoolkit.version}"
            echo "CUTLASS: ${cudaPackages.cutlass.version}"
            echo "CUTLASS: ${cudaPackages.cutlass}"
          '';
        };
      }
    );
}
