CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

SRC_DIR = src
OBJ_DIR = build

SOURCES = $(SRC_DIR)/pauli.cpp $(SRC_DIR)/tests.cpp $(SRC_DIR)/main.cpp
OBJECTS = $(OBJ_DIR)/pauli.o $(OBJ_DIR)/tests.o $(OBJ_DIR)/main.o

TARGET = pauli_sim

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
