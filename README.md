# portal-particle-extraction
Particle extraction from cryoET data portal tilt series.

# Example run:
```
time python /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/pyrelion-runs/data_portal_subtomo_extract.py \
  --particles-starfile /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/input/session1_polnet_0_ribosome.star \
  --tiltseries-dir /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/input/tiltSeries \
  --tiltseries-starfile /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/input/tiltSeries/aligned_tilt_series.star \
  --aln-dir /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/aretomo_mock/session1/run001/ \
  --output-dir /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/relion_mock/Extract/mockjob001/ \
  --particles-tomo-name-prefix "session1_" \
  --particle-diameter 330 --tiltseries-pixel-size 10.0 --tiltseries-x 630 --tiltseries-y 630 --box-size 64 --bin 1 --debug &> relion_mock/mock_subtomo_extract.log
```
