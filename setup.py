from setuptools import setup,find_packages


hypen= '-e .'

def get_requirements(file_path):
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if hypen in requirements:
            requirements.remove(hypen)
    return requirements


setup(
    name = 'facemask_detection_project',
    version= '0.1',
    author= 'amballa.mahesh',
    author_email='amballa.mahesh89@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')

)