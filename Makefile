BUILD_DIR := build

.PHONY: build mbuild run clean all

build:
	@cmake -S . -B $(BUILD_DIR) -G Ninja
	@cmake --build $(BUILD_DIR)

mbuild:
	@meson setup $(BUILD_DIR) || true
	@meson compile -C $(BUILD_DIR)

run:
	@./$(BUILD_DIR)/ringtest

clean:
	@rm -rf $(BUILD_DIR)

all: build run
