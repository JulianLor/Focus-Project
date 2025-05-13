# Focus Project | Medical Microsystems Laboratory (ETH)

This repository served to assist the development of a remote magnetic actuation system for medical biohybrid robots in a potential glioblastome therapy.
It includes simulations and visualisation that were progressively implemented and their insights were used in for this actuation system. This was part 
of a D-HEST Focus Project 2024-2025 conducted at ETH.

## Content

Within the repository the /Archive folder includes an older but still functional version on how the electromagnetic simulations were simulated. They
are still used to generate the animation. The newer versions are responsible for creating all the other functionlities of this code. Only modern code
is tested in the test section.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/JulianLor/Focus-Project
cd Focus-Project
``` 
2. Install dependencies:
```bash
pip install -r requirements.txt  
```

## Config files

The main config file (path: src.main.resources.config.py) holds all global variables used in both the older versions (denoted as 'old animation gen')
and the new object oriented part of the code. Descriptions on functionality are included as comments.

## Features

All the key features are included as functions in the main.py (path: src.main.python.main.py) and include:

- Animation generation: 3D time dependent vector field animation
- Volume analysis: RMF & Canc Magnitude and defining points of rotation (Including csv. export of RMF, Canc, Is.rot data)
- Current analysis: Power usage and visualisation on singular current usage
- Force analysis: Internal interactions of electromagnets and permanent magnets

## To-Do

This repository fulfilled its purpose within the Focus Project. Given the inaccuracy of other software this code could be expanded to serve this purpose

- Generalisation of principles to create 3D simulation software.
- Addition of Maxwell's Laws for more accurate electromagnet modelling

## Contributing guidelines

Pull requests and reporting of bugs will be handled by JulianLor. Contact is written below.

## License

License type is stated in the License file

## Credits

Support accredited by the Medical Microsystems Laboratory at ETH Zurich.

## Contact

E-mail: julian@zukunft.com



