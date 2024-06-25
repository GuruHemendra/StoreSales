from setuptools import setup,find_packages

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path :str):

    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
    
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name = "Store Sales",
    author_email="guruhemendraputhuru@gmail.com",
    version= "0.0.1",
    author="Hemendra",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)