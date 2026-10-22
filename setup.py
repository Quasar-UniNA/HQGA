from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()
    
setup(
  name = 'HQGA',         
  packages = ['HQGA'],   
  version = 'v0.4.0',      
  license='MIT',     
  description = 'A library for implementing Hybrid Quantum Genetic Algorithm (HQGA)',  
  long_description=readme,
  long_description_content_type="text/markdown",
  author = 'Autilia Vitiello',                  
  author_email = 'autilia.vitiello@unina.it',      
  url = 'https://github.com/Quasar-UniNA/HQGA',   
  keywords = ['Optimization Algorithms', 'Quantum Computing', 'Evolutionary Algorithms'],   
  install_requires=[            
'matplotlib>=3.10.3', 
'numpy>=2.5.1',
'openpyxl>=3.1.5',
'qiskit>=2.5.2',
'qiskit_ibm_runtime>=0.48.0',
'sympy>=1.14.0',
'tqdm>=4.65.0'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3'
  ],
)
