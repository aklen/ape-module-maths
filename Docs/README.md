# Maths

**Ape.Module.Maths** is an optional Ape module: discrete linear Kalman filtering, Mahalanobis distance, a chi-square gate threshold, rectangular min-cost assignment, and a constant-velocity transition. The caller supplies every matrix, every probability, and every noise scale.

It is not part of Core. Core stays the scene, replica, and plugin host. This module is a library you add when a process needs those steps. It does not register a host service or a scene entity. Geodesy stays in Geo.

## Place in the ecosystem

| Piece | Role |
| --- | --- |
| [ape-skeleton](https://github.com/aklen/ape-skeleton) | Workspace vessel (`./ape sync`, `./ape build`) |
| [ape-core](https://github.com/aklen/ape-core) | Framework: scene, DI, plugins, commits |
| [ape-launcher](https://github.com/aklen/ape-launcher) | Host executable |
| **this repo** | Linear estimation and assignment |

After `./ape sync` the tree looks like:

```
ape-skeleton/
└── src/
    ├── Ape.Core/
    ├── Ape.Launcher/
    └── Ape.Modules/
        └── Ape.Module.Maths/   ← this repository
```

Enable it in `workspace.yaml` under `modules:` (`name: Ape.Module.Maths`, this GitHub URL). Other projects reference `Ape.Module.Maths.csproj`. There is no host JSON section.

`MathsModuleIds.ModuleId` and `MathsModule.Id` are `Ape.Module.Maths`.

## Kalman filter

`KalmanFilter.Predict` applies a caller-supplied transition and process-noise matrix:

- mean ← F mean
- covariance ← F P Fᵀ + Q, then symmetrized

`KalmanFilter.Update` applies a linear measurement. The update is the Joseph form, and the covariance is symmetrized again. The returned `MahalanobisSquared` is yᵀ S⁻¹ y, where y is the innovation and S is its covariance. A covariance that is not positive definite throws.

`KalmanState.Create` takes the mean and a square covariance of the same length.

## Distance and gate

`Mahalanobis.Squared` and `Mahalanobis.Distance` take a residual and its covariance.

`ChiSquare.Threshold(degreesOfFreedom, probability)` is the chi-square quantile at that probability. Compare it with a squared Mahalanobis distance. The probability is an argument on each call. One degree of freedom is the squared normal quantile. Two degrees of freedom are −2 ln(1 − probability).

## Assignment

`Hungarian.Minimize(cost, rows, columns)` assigns rows to columns at minimum cost. The cost array is row-major. A non-finite entry is an absent pair and stays unmatched. The solver uses as many finite pairs as it can, and among those matchings it minimizes the sum. Rows and columns may differ in count. The result lists the matched pairs, ordered by row.

## Constant velocity

`ConstantVelocity` builds the transition and the process-noise matrix for independent axes. The state is positions, then velocities: index `i` is the position of axis `i`, and index `axes + i` is its velocity.

`Transition(seconds, spatialDimension)` is identity, with `seconds` on each position-from-velocity entry.

`ProcessNoise(seconds, noiseDensity, spatialDimension)` is continuous white acceleration. For each axis, with time step dt and density q:

| Entry | Value |
| --- | --- |
| position, position | q dt³ / 3 |
| position, velocity | q dt² / 2 |
| velocity, velocity | q dt |

Axes do not couple. `noiseDensity` is the caller's scale. A zero time step returns an identity transition and a zero process-noise matrix.

## License

Copyright (c) 2026 [Akos Hamori](https://github.com/aklen).

Licensed under the [Mozilla Public License 2.0 (MPL-2.0)](https://www.mozilla.org/MPL/2.0/).
