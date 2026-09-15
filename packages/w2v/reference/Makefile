CC ?= cc
CPPFLAGS := -Iinclude -Iinternal
CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -Werror -O2 -MMD -MP
LDLIBS := -lm -pthread
SOURCES := \
	src/config/config.c \
	src/corpus/corpus.c \
	src/corpus/tokenizer.c \
	src/model/atomic_float.c \
	src/model/model.c \
	src/model/sigmoid_table.c \
	src/random/rng.c \
	src/training/cbow.c \
	src/training/context.c \
	src/training/hierarchical_softmax.c \
	src/training/negative_sampling.c \
	src/training/objective.c \
	src/training/objective_math.c \
	src/training/skip_gram.c \
	src/training/trainer.c \
	src/training/worker.c \
	src/vocab/huffman.c \
	src/vocab/negative_sampler.c \
	src/vocab/vocab.c
OBJECTS := $(SOURCES:.c=.o)
DEPENDENCIES := $(OBJECTS:.o=.d)
TEST_DEPENDENCIES := tests/test_w2v.d tests/test_public_api.d

-include $(DEPENDENCIES) $(TEST_DEPENDENCIES)

.PHONY: all test sanitize asan-ubsan tsan clean

all: libw2v.a

libw2v.a: $(OBJECTS)
	$(AR) rcs $@ $^

tests/test_w2v: tests/test_w2v.c libw2v.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< libw2v.a $(LDLIBS) -o $@

tests/test_public_api: tests/test_public_api.c libw2v.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< libw2v.a $(LDLIBS) -o $@

test: tests/test_w2v tests/test_public_api
	./tests/test_w2v
	./tests/test_public_api

asan-ubsan:
	$(CC) $(CPPFLAGS) -std=c11 -Wall -Wextra -Wpedantic -Werror -g -fsanitize=address,undefined $(SOURCES) tests/test_w2v.c $(LDLIBS) -o tests/test_w2v_sanitize
	ASAN_OPTIONS=detect_leaks=1 ./tests/test_w2v_sanitize

tsan:
	$(CC) $(CPPFLAGS) -std=c11 -Wall -Wextra -Wpedantic -Werror -g -fsanitize=thread $(SOURCES) tests/test_w2v.c $(LDLIBS) -o tests/test_w2v_tsan
	TSAN_OPTIONS=halt_on_error=1 ./tests/test_w2v_tsan

sanitize: asan-ubsan tsan

clean:
	$(RM) $(OBJECTS) $(DEPENDENCIES) $(TEST_DEPENDENCIES) libw2v.a tests/test_w2v tests/test_public_api tests/test_w2v_sanitize tests/test_w2v_tsan
