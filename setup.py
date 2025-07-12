from setuptools import find_packages, setup

setup(
    name="DSV",
    version="0.1",
    description="DSV Project",
    author="Your Name",
    packages=find_packages(where=".", include=["DSV*"]),
    python_requires=">=3.11",
    include_package_data=True,
    zip_safe=False,
)
