from setuptools import setup, find_packages
from typing import List  

__version__ = "0.0.1"
SRC_REPO = "us_visa"

setup(
    name=SRC_REPO,
    version=__version__,
    description = "This is a ML classification project to predict whether US Visa application is certified or denied",
    author="Aadarsh Kushwaha",
    author_email="aadarshkushwaha0208@gmail.com",
    packages=find_packages()
)

'''
Sometimes -e . in requiremets.txt not work at that that use code given below


HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    this function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name=SRC_REPO,
    version=__version__,
    description = "This is a ML classification project to predict whether US Visa application is certified or denied",
    author="Aadarsh Kushwaha",
    author_email="aadarshkushwaha0208@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

'''