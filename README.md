# Shooting_Shapes

Bozic and Svetina [10.1007/s00249-004-0404-5](https://link.springer.com/article/10.1007/s00249-004-0404-5) vesicle growth using Seifert et al. [10.1103/PhysRevA.44.1182](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.44.1182) style shooting:
sphere → prolate → pear → thin-neck two-spheres.

## Run

```bash
cd Shooting_Shapes
python run_trajectory.py
```
## Vesicle growth notes

- **B**: sphere to prolate transition based on eq. (9): `τ_B` from `c₀,cr(η)`.
- **D**: branch switch around `τ ≳ 0.95` where pear energy drops below prolate at the same `(v̄, c₀)`.
- **Thin-neck shooting failure**: Approximate as two spheres, and assume pressure continues to increase linearly to `τ = 1`.

https://github.com/user-attachments/assets/a1618bfa-e6ff-47f2-b138-a9d3a2a6da68

<img width="2175" height="1200" alt="full_trajectory" src="https://github.com/user-attachments/assets/8f6ab613-e107-4d46-b587-b0760af7a9de" />
