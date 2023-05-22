# HQGA [![Made at Quasar!](https://img.shields.io/badge/Unina-%20QuasarLab-blue)](http://quasar.unina.it) [![Made at Quasar!](https://img.shields.io/badge/Documentation-%20Readthedocs-brightgreen)](https://hqga.readthedocs.io/en/latest/index.html) [![Made at Quasar!](https://img.shields.io/badge/Related-%20Paper-orange)](https://www.sciencedirect.com/science/article/abs/pii/S002002552100640X)

This repo contains the code for executing Hybrid Quantum Genetic Algorithm (HQGA) proposed in:

**''G. Acampora and A. Vitiello, "Implementing evolutionary optimization on actual quantum processors,"
    in Information Sciences, 2021, doi: 10.1016/j.ins.2021.06.049.''**
    
    
## How to install

The package can be installed with Python's pip package manager.

```bash
pip install HQGA
```

# Example
This is a basic example to use HQGA for solving Sphere problem on Qasm Simulator.

```python
from HQGA import problems as p, hqga_utils, utils, hqga_algorithm
from HQGA.utils import computeHammingDistance

from qiskit import Aer
import math

simulator = Aer.get_backend('qasm_simulator')
device_features= hqga_utils.device(simulator, False)

params= hqga_utils.ReinforcementParameters(3, 5, math.pi / 16, math.pi / 16, 0.3)
params.draw_circuit=True

problem = p.SphereProblem(num_bit_code=5)

circuit = hqga_utils.setupCircuit(params.pop_size, problem.dim * problem.num_bit_code)

gBest, chromosome_evolution,bests = hqga_algorithm.runQGA(device_features, circuit, params,problem)

dist=computeHammingDistance(gBest.chr, problem)
print("The Hamming distance to the optimum value is: ", dist)
utils.writeBestsXls("Bests.xlsx", bests)
utils.writeChromosomeEvolutionXls("ChromosomeEvolution.xlsx", chromosome_evolution)
```


## Credits

Please cite the work using the following Bibtex entry:

```text
@article{ACAMPORA2021542,
title = {Implementing evolutionary optimization on actual quantum processors},
journal = {Information Sciences},
volume = {575},
pages = {542-562},
year = {2021},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2021.06.049},
url = {https://www.sciencedirect.com/science/article/pii/S002002552100640X},
author = {Giovanni Acampora and Autilia Vitiello}
}

```
