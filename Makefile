BUILD_DIR := build

.PHONY: build clean all

build:
	@cmake -S . -B $(BUILD_DIR) -G Ninja
	@cmake --build $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)

all: build run

