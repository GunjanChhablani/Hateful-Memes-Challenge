.PHONY: clean all requirements setup

# GLOBALS

ENVIRONMENT = env.yaml
ENV_NAME = hmc

ifeq (,$(shell which conda))
HAS_CONDA = False
else
HAS_CONDA = True
endif

#
all:create_environment requirements setup

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


requirements:
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt

setup:
	python setup.py build
	python setup.py install

create_environment:
ifeq (True,$(HAS_CONDA))
ifeq ($(ENV_NAME),$(findstring $(ENV_NAME),$(shell conda env list)))
	@echo "Environment already exists. Trying to update."
	conda env update -f $(ENVIRONMENT)
else
	conda env create -f $(ENVIRONMENT)
endif
else
	@echo "Unable to find conda. Make sure conda is installed and try again."
endif

#help:
