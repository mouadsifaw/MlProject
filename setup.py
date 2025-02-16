from setuptools import setup, find_packages
from typing import List

def get_requirements(path:str)->List[str]:
    requirements = []
    hyphen_dot_e = '-e .'
    with open('requirements.txt') as file_object:
        requirements = file_object.readlines()
        requirements = [requirement.replace('\n','') for requirement in requirements]

        if hyphen_dot_e in requirements:
            requirements.remove(hyphen_dot_e)
    return requirements
 



setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'MouadSifaw',
    author_email='mouaduzumaki2001@gmail.com',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)

