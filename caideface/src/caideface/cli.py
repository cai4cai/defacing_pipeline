"""Command-line interface for caideface."""

import argparse
import logging
import sys

from .pipeline import DefacePipeline
from .reorient import reorient_batch
from .skull_strip import skull_strip_batch, get_default_device
from .register import deface_batch
from .anonymize import anonymize_batch, anonymize_single, load_ner_model, generate_fake_names


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    # Shared parent so -v works on any subcommand
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    parser = argparse.ArgumentParser(
        prog="caideface",
        description="MRI defacing pipeline from cai4cai: reorientation, skull-stripping, and affine-based defacing.",
        parents=[parent],
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run: full pipeline ---
    run_parser = subparsers.add_parser("run", help="Run the full defacing pipeline", parents=[parent])
    run_parser.add_argument("input_dir", help="Directory containing raw NIfTI files")
    run_parser.add_argument("output_dir", help="Root output directory")
    run_parser.add_argument("--modality", required=True, choices=["mri", "ct"], help="Image modality (required)")
    run_parser.add_argument("--brainsfit", required=True, help="Path to BRAINSFit executable")
    run_parser.add_argument("--brainsresample", required=True, help="Path to BRAINSResample executable")
    run_parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Device for HD-BET (auto-detected if omitted)")
    run_parser.add_argument("--no-tta", action="store_true", default=True, help="Disable HD-BET test-time augmentation (default: disabled)")
    run_parser.add_argument("--dilation-mm", type=float, default=14.0, help="Brain mask dilation in mm (default: 14)")
    run_parser.add_argument("--background", type=float, default=None, help="Background value for defaced voxels (auto-detected per volume if omitted)")
    run_parser.add_argument("--template", default=None, help="Custom MNI152 skull-stripped template (uses bundled if omitted)")
    run_parser.add_argument("--face-mask", default=None, help="Custom face mask in MNI152 space (uses bundled if omitted)")
    run_parser.add_argument("--steps", default="all", help="Steps to run: all, or comma-separated: reorient,skull_strip,deface")

    # --- reorient ---
    reorient_parser = subparsers.add_parser("reorient", help="Step 1: Reorient NIfTI scans to RAS", parents=[parent])
    reorient_parser.add_argument("input_dir", help="Directory with NIfTI files")
    reorient_parser.add_argument("output_dir", help="Output directory for reoriented files")

    # --- skull-strip ---
    ss_parser = subparsers.add_parser("skull-strip", help="Step 2: Skull-strip with HD-BET", parents=[parent])
    ss_parser.add_argument("input_dir", help="Directory with reoriented NIfTI files")
    ss_parser.add_argument("output_dir", help="Output directory for skull-stripped results")
    ss_parser.add_argument("--modality", required=True, choices=["mri", "ct"], help="Image modality (required)")
    ss_parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Device for HD-BET")
    ss_parser.add_argument("--no-tta", action="store_true", default=True, help="Disable test-time augmentation")
    ss_parser.add_argument("--dilation-mm", type=float, default=14.0, help="Dilation in mm")

    # --- deface ---
    deface_parser = subparsers.add_parser("deface", help="Step 3: Register and deface", parents=[parent])
    deface_parser.add_argument("reoriented_dir", help="Directory with reoriented scans (Step 1 output)")
    deface_parser.add_argument("skullstripped_dir", help="Directory with skull-stripped results (Step 2 output)")
    deface_parser.add_argument("output_dir", help="Output directory for defaced scans")
    deface_parser.add_argument("--modality", required=True, choices=["mri", "ct"], help="Image modality (required)")
    deface_parser.add_argument("--brainsfit", required=True, help="Path to BRAINSFit executable")
    deface_parser.add_argument("--brainsresample", required=True, help="Path to BRAINSResample executable")
    deface_parser.add_argument("--template", default=None, help="Custom MNI152 skull-stripped template")
    deface_parser.add_argument("--face-mask", default=None, help="Custom face mask in MNI152 space")
    deface_parser.add_argument("--background", type=float, default=None, help="Background value (auto-detected per volume if omitted)")

    # --- anonymize (batch) ---
    anon_parser = subparsers.add_parser("anonymize", help="Anonymize personal names in all .txt files in a directory", parents=[parent])
    anon_parser.add_argument("input_dir", help="Directory containing .txt files")
    anon_parser.add_argument("output_dir", help="Output directory for anonymized files")
    anon_parser.add_argument("--model", default=None, help="Custom NER model directory (uses bundled if omitted)")
    anon_parser.add_argument("--n-names", type=int, default=50, help="Size of fake name pool (default: 50)")
    anon_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # --- anonymize-single ---
    anon_single_parser = subparsers.add_parser("anonymize-single", help="Anonymize personal names in a single .txt file", parents=[parent])
    anon_single_parser.add_argument("input_file", help="Input .txt file")
    anon_single_parser.add_argument("output_file", help="Output file path")
    anon_single_parser.add_argument("--model", default=None, help="Custom NER model directory (uses bundled if omitted)")
    anon_single_parser.add_argument("--n-names", type=int, default=50, help="Size of fake name pool (default: 50)")
    anon_single_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    _setup_logging(args.verbose)

    if args.command == "run":
        pipeline = DefacePipeline(
            brainsfit_path=args.brainsfit,
            brainsresample_path=args.brainsresample,
            modality=args.modality,
            device=args.device,
            disable_tta=args.no_tta,
            desired_dilation_mm=args.dilation_mm,
            background_value=args.background,
            target_path=args.template,
            face_mask_path=args.face_mask,
        )
        results = pipeline.run(args.input_dir, args.output_dir, steps=args.steps)
        failed = results.get("failed_defacing", [])
        if failed:
            print(f"\n{len(failed)} scan(s) failed to deface. See output log for details.")
            sys.exit(1)

    elif args.command == "reorient":
        reorient_batch(args.input_dir, args.output_dir)

    elif args.command == "skull-strip":
        skull_strip_batch(
            args.input_dir,
            args.output_dir,
            modality=args.modality,
            device=args.device,
            disable_tta=args.no_tta,
            desired_dilation_mm=args.dilation_mm,
        )

    elif args.command == "deface":
        failed = deface_batch(
            reoriented_dir=args.reoriented_dir,
            skullstripped_dir=args.skullstripped_dir,
            output_dir=args.output_dir,
            brainsfit_path=args.brainsfit,
            brainsresample_path=args.brainsresample,
            modality=args.modality,
            target_path=args.template,
            face_mask_path=args.face_mask,
            background_value=args.background,
        )
        if failed:
            print(f"\n{len(failed)} scan(s) failed. See output log.")
            sys.exit(1)

    elif args.command == "anonymize":
        log_df = anonymize_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_dir=args.model,
            n_fake_names=args.n_names,
            seed=args.seed,
        )
        total = log_df["replacements"].sum()
        print(f"\nAnonymized {len(log_df)} file(s), {total} name(s) replaced.")

    elif args.command == "anonymize-single":
        nlp = load_ner_model(args.model)
        fake_names = generate_fake_names(n=args.n_names, seed=args.seed)
        result = anonymize_single(args.input_file, args.output_file, nlp, fake_names)
        names = result["names_found"]
        print(f"\nAnonymized {result['replacements']} name(s): {names}")
        print(f"Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
