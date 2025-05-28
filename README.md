# MagneticLevitationMPC

## 🚀 Quick Start

### Prerequisites

If you don't have [uv](https://github.com/astral-sh/uv) (a fast Python package installer and resolver) installed, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Running the Simulation

To run the standard simulation with full stabilization:

```bash
uv run main.py
```

The plots and animation will be saved to `gfx/MagneticLevitationMPC/`.

To run the simulation with only the backstepping controller (without switching to PD control):

```bash
uv run main.py --backstepping-only
```

This alternative simulation output will be saved to `gfx/MagneticLevitationMPC/`.

## 🙏 Authors
* [Egor Miroshnichenko](https://github.com/Chenkomirosh)
* [Anton Bolychev](https://github.com/antonbolychev)
* [Vladislav Sarmatin](https://github.com/VladSarm)
* [Arsenii Shavrin](https://github.com/ArseniiSh)