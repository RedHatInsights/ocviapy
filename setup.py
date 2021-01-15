from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ocviapy",
    use_scm_version=True,
    description=("Wraps OpenShift shell utility (oc)"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    author="Brandon Squizzato",
    author_email="bsquizza@redhat.com",
    url="https://www.github.com/RedHatInsights/ocviapy",
    packages=find_packages(),
    keywords=["openshift", "kubernetes"],
    setup_requires=["setuptools_scm"],
    include_package_data=True,
    install_requires=[
        "sh>=1.13.1",
        "pyyaml",
        "wait_for",
        "kubernetes",
        "anytree",
    ],
    classifiers=[
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    python_requires=">=3.6",
)
