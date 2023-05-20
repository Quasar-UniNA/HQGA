from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()
    
setup(
  name = 'HQGA',         
  packages = ['HQGA'],   
  version = '0.1.0',      
  license='MIT',     
  description = 'A library for implementing HQGA',  
  long_description=readme,
  author = 'Autilia Vitiello',                  
  author_email = 'autilia.vitiello@unina.it',      
  url = 'https://github.com/Quasar-UniNA/HQGA',   
  keywords = ['Optimization Algorithm', 'Quantum Computing', 'Evolutionary Algorithms'],   
  install_requires=[            
'matplotlib', 
'numpy',
'openpyxl==3.1.2',
'qiskit==0.42.0',
'qiskit_aer==0.12.0',
'qiskit_ibmq_provider==0.20.2',
'qiskit_terra==0.23.2',
'sympy',
'tqdm'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3'
  ],
)
