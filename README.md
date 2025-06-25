<h1 align="center">
<img src="./res/logo2.png" width="400">
</h1><br>

[![Python
3.6+](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-360/)


**Precision at the core: total tolerance stack‑up, intelligently fused.**

StackCore is a lightweight Python package for mechanical **tolerance stack‑up analysis**, integrating Monte Carlo simulation and data‑driven intelligence.

---

## 🚀 Features

- 🧮 **Tolerance stack‑up under control**: manage multi‑component linear and geometric assemblies.
- 🔁 **Monte Carlo engine**: generate distributions, confidence ranges, and worst‑case scenarios.
- 🧠 **Subtle intelligence** (coming soon!): leverage surrogate modelling for quick inference responses at comparable convergence.
- 📊 **Flexible output**: summary stats, histograms, and data visualization tools.
- 🔌 **Clean APIs**: `Stack`, `PStack`
- 🛠️ **Designed for engineers**: zone‑independent, boundary‑aware, CAD/tool‑agnostic.

---

## 📥 Installation

```bash
pip install stackcore
```

## Sample Use
To import the ```stackcore``` module, use
```python
import stackcore
```

The main component, the ```Stack``` can be imported as follows
```python
from stackcore import Stack
```

The ```Stack``` Component taks a few key arguments:
### Main Plane
The main plane is defined by a set of 3 points, and defines the main plane that the tolerance stack up will measure against. Here's an example:
```python
m=np.array([[3,  2, -1],
            [0,  3,  3],
            [-1, 2,  4]])
```
### Components
The components variable contains a dictioary of planes and their respective tolerances, with many various tolerance types such as displacement and radial (axis).
Here's an example:
```python
components = [
             {'plane': np.array([[-8, 1, 5],
                                 [-4, 7, 3],
                                 [-3, 2, 6]]),
              'tolerances': [{'type': 'cylindrical',
                              'tol': [-0.1, +0.1],
                              'axis': np.array([[0,0,1], 
                                                [0,0,1],
                                                [-.7, -0.9, 0.]])},
                              {'type': 'displacement',
                               'tol' : [-0.1, +0.1],
                               'axis': np.array([[0,0,1], 
                                                 [0,0,1],
                                                 [-.7, -0.9, 0.]])}]}
]
```
### Metrics
The metrics variable contains a dictionary of the targeted metrics and their reference type. Here's an example:
```python
metrics = [
    { 'name': 'alpha',
      'type': 'angle', 
      'ref': np.array([0,0,1])
    }
]
```
### Path
The path variable defines where any generated figures should be stored. Here's an example:
```python
path = 'path/to/figures/'
```
<br>

## Parallel Processing
For faster processing, the ```PStack``` object is provided that uses the ```numba``` package to parallelize the loops in the monte carlo simulation. To use the ```PStack```, import it using
```python
from stackcore import PStack
```
All of the same functions of the ```Stack``` object are available in the ```PStack``` Object.

For a full example, see [example.py](example.py)