from setuptools import find_packages,setup

DISTNAME = "core"
AUTHOR = 'Gunjan Chhablani'
AUTHOR_EMAIL = 'chhablani.gunjan@gmail.com'
#EXCLUDES =
#DEPENDENCY_LINKS

if __name__=="__main__":
    setup(name=DISTNAME,packages=find_packages(),author = AUTHOR, author_email = AUTHOR_EMAIL)
