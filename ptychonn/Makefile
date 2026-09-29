PYBIND11_INCLUDE:=$(shell python3.11 -m pybind11 --includes)
PYTHON_SHARED_LIB_EXTENSION:=$(shell python3.11-config --extension-suffix)

all: unipipe_scheduler${PYTHON_SHARED_LIB_EXTENSION}

unipipe_scheduler${PYTHON_SHARED_LIB_EXTENSION}: unipipe_scheduler.cpp
	g++ -O3 -Wall -shared -std=c++11 -fPIC ${PYBIND11_INCLUDE} $^ -o $@

clean:
	rm unipipe_scheduler${PYTHON_SHARED_LIB_EXTENSION}