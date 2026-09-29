CC ?= cc
CFLAGS := -std=c11 -O3 -march=native -g -fno-omit-frame-pointer -pthread -D_GNU_SOURCE
CPPFLAGS := -I../reference/include -I../reference/internal
LDLIBS := -lm -pthread
BUILD := build
REFERENCE_SOURCES := $(wildcard ../reference/src/*/*.c)

.PHONY: all clean

all: $(BUILD)/original-w2v $(BUILD)/reference-w2v $(BUILD)/rust-w2v

$(BUILD):
	mkdir -p $@

$(BUILD)/original-w2v: ../reference/z_original_w2v.c | $(BUILD)
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

$(BUILD)/reference-w2v: reference_main.c $(REFERENCE_SOURCES) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/rust-w2v: rust-runner/src/main.rs rust-runner/Cargo.toml ../Cargo.toml $(wildcard ../src/*.rs ../src/training/*.rs) | $(BUILD)
	RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" cargo build --manifest-path rust-runner/Cargo.toml --release --target-dir $(BUILD)/rust-target
	cp $(BUILD)/rust-target/release/w2v-profile $@

clean:
	$(RM) -r $(BUILD)
