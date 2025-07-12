
CXX = g++
CONDA_PREFIX ?= $(shell echo $$CONDA_PREFIX)

# ensure the correct conda env (python 3.11) prefix 
$(info Current Conda Prefix: $(CONDA_PREFIX))

PYTHON_INCLUDE = -I$(CONDA_PREFIX)/include/python3.11
PYBIND11_INCLUDE = -I$(CONDA_PREFIX)/lib/python3.11/site-packages/pybind11/include
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC $(PYTHON_INCLUDE) $(PYBIND11_INCLUDE)

TARGET = hybrid_optimizer.cpython-311-x86_64-linux-gnu.so

SRC = solver.cc gpu_database.cpp hardware_config.cpp

PYTHON_LIB = -L$(CONDA_PREFIX)/lib/python3.11/config-3.11-x86_64-linux-gnu -L$(CONDA_PREFIX)/lib  -lpthread -ldl  -lutil -lm 

SITE_PACKAGES = $(shell python -c "import site; print(site.getsitepackages()[0])")

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(PYTHON_LIB)

install: $(TARGET)
	@echo "Installing to: $(SITE_PACKAGES)"
	@cp $(TARGET) $(SITE_PACKAGES)/
	@echo "Module installed successfully!"
	@echo "You can now import it anywhere with: import hybrid_optimizer"

uninstall:
	@rm -f $(SITE_PACKAGES)/$(TARGET)
	@echo "Module uninstalled"

test-import:
	@python -c "import hybrid_optimizer; print('Import successful!')"

clean:
	@rm -f $(TARGET) *.o

.PHONY: all install uninstall test-import clean