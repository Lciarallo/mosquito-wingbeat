"""Compatibility entry point for the corrected species calibration experiment.

The historical optimiser used OOF calibration predictions whose base models
included the outer test fold in their fit. Its source is preserved in Git at
1ab88b8; its numerical artifacts remain historical and are not overwritten here.

This command now runs the isolated three-member fit/calibration/test experiment.
Use --output to select a new directory; existing protocols cannot be overwritten.
"""
from nested_species_ensemble import main


if __name__ == '__main__':
    main()
