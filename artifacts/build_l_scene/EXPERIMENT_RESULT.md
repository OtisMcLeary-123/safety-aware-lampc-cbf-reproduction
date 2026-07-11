## Material Passport

- Material ID: `safe-panda-build-l-seed-7`
- Type: deterministic simulator scene render
- Verification status: `VERIFIED`
- Environment: `PandaBuildL-v3`
- Safe Panda commit: `c2c2bae9ee0b738fd7c5a5f6259a3a37da95718c`

## Result

The restored Gymnasium environment creates four movable colored cubes and four translucent target cubes arranged as an L. The scene reset, zero-action step, observation-space validation, and 720x480 headless render pass.

The render and `scene.json` were reproduced from a newly created virtual environment. The PNG SHA-256 matched exactly:

`805e44125acdb299d549ba71d63a8246c12df8619e6d3a542c8a9f2c1b829645`

## Rendering limitation

The reference image was produced by the legacy GUI/OpenGL path. The verified artifact uses PyBullet TinyRenderer for deterministic headless execution. Consequently, camera framing is similar, but the sky/background and target transparency are not pixel-identical to the reference.

## Command

```bash
python scripts/render_build_l_scene.py --seed 7 \
  --output-dir artifacts/build_l_scene
```
