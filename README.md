# nwkpy

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](docs/)

**nwkpy** is a Python library for calculating electronic band structures of semiconductor nanowires using the 8-band k·p method with finite element discretization. The library specializes in core-shell heterostructures with self-consistent Schrödinger-Poisson coupling and advanced numerical techniques for computational efficiency and accuracy.

## Key Features

- **8-band k·p Hamiltonian**: Complete treatment of conduction and valence bands including spin-orbit coupling
- **Finite Element Method (FEM)**: Flexible spatial discretization with FreeFem++ integration
- **Core-Shell Nanowires**: Specialized support for hexagonal cross-section heterostructures  
- **Self-Consistent Coupling**: Schrödinger-Poisson equations with Broyden mixing for rapid convergence
- **Advanced Numerics**: Spurious solution suppression, Modified Envelope Function Approximation (EFA)
- **High Performance**: MPI parallelization, optimized sparse matrix solvers, inhomogeneous mesh support
- **Comprehensive Analysis**: Band structure visualization, charge density plots, electrostatic potential mapping

## Physics Background

nwkpy implements state-of-the-art methods for semiconductor nanostructure calculations:

- **k·p Theory**: 8-band Kane model with P-parameter rescaling to eliminate spurious solutions
- **Heterostructure Physics**: Type I/II band alignments, broken-gap structures, carrier localization
- **Electrostatics**: Self-consistent treatment of built-in fields, external electric fields, charge redistribution
- **Material Systems**: Comprehensive database of III-V semiconductors (InAs, GaAs, GaSb, InP, etc.)

## Quick Start

### Installation

```bash
git clone https://github.com/your-username/nwkpy.git
cd nwkpy
pip install -e .
```

### Basic Workflow

nwkpy calculations follow a three-step workflow:

```bash
# 1. Generate finite element mesh
python scripts/mesh_generation/main.py

# 2. Calculate band structure (non-self-consistent)
python scripts/band_structure/main.py

# 3. Self-consistent calculation with parameter sweeps
python scripts/self_consistent/main.py
```

### Simple Example

```python
import numpy as np
from nwkpy.fem import Mesh
from nwkpy import BandStructure

# Load pre-generated mesh
mesh = Mesh(mesh_name="hexagonal_nanowire.msh", 
           reg2mat={1: "InAs", 2: "GaSb"})

# Set up k-space sampling
kz_values = np.linspace(0, 0.05, 20) * np.pi / np.sqrt(3) / 6.0583

# Calculate band structure
bs = BandStructure(
    mesh=mesh,
    kzvals=kz_values,
    valence_band_edges={"InAs": 0.0, "GaSb": 0.56},
    temperature=4.0,
    number_eigenvalues=20
)
bs.run()

# Visualize results
bs.plot_bands()
```

## Core Components

### 1. Mesh Generation
- **Purpose**: Create optimized finite element grids for nanowire cross-sections
- **Geometry**: Hexagonal symmetry with core-shell regions
- **Output**: FreeFem++ compatible mesh files with material region definitions

### 2. Band Structure Solver  
- **Method**: 8-band k·p Hamiltonian with FEM spatial discretization
- **Features**: MPI parallelization over k-points, spurious solution suppression
- **Analysis**: Carrier character classification, regional charge distribution

### 3. Self-Consistent Solver
- **Coupling**: Schrödinger-Poisson equations with iterative solution
- **Convergence**: Broyden mixing for accelerated convergence
- **Applications**: Multi-parameter sweeps, external field effects

## Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)**: Get up and running in 5 minutes
- **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup including FreeFem++ integration  
- **[Tutorial Collection](docs/TUTORIALS/)**: Step-by-step examples from basic to advanced
- **[API Reference](docs/API_REFERENCE.md)**: Complete class and method documentation
- **[Physics Background](docs/PHYSICS_BACKGROUND.md)**: Theoretical foundations and numerical methods

## Example Applications

- **InAs/GaSb Core-Shell Nanowires**: Broken-gap heterostructures for topological applications
- **External Electric Fields**: Band structure engineering and carrier localization control  
- **Temperature-Dependent Properties**: Thermal effects on electronic structure
- **Multi-Material Systems**: Complex heterostructure with multiple interfaces

## Requirements

- **Python**: 3.7+
- **Core Dependencies**: NumPy, SciPy, matplotlib
- **MPI Support**: mpi4py for parallel calculations
- **External Tools**: FreeFem++ for advanced mesh generation
- **Optional**: Jupyter for interactive analysis

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/DEVELOPMENT/contributing.md) for:

- Code style and standards
- Testing procedures  
- Documentation requirements
- Issue reporting and feature requests

## Citation

If you use nwkpy in your research, please cite:

```bibtex
@software{nwkpy2025,
  title={nwkpy: 8-band k·p calculations for semiconductor nanowires},
  author={Vezzosi, A. and Goldoni, G.},
  year={2025},
  url={https://github.com/your-username/nwkpy}
}
```

## License

nwkpy is released under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- **Documentation**: [https://nwkpy.readthedocs.io](https://nwkpy.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-username/nwkpy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/nwkpy/discussions)

---

**Developed by the Computational Nanostructures Group**  
*A. Vezzosi, G. Goldoni*