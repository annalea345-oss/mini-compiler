/* Included in generated programs: a JSON reader for named, fixed-shape tensors.
 * Only objects of numeric vectors/matrices are accepted, not general JSON values.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
  const char *name;
  TensorValue *values;
  int rank;
  size_t shape[2];
  int seen;
} TensorInput;

_Noreturn static void fail(const char *input, const char *message) {
  if (input) fprintf(stderr, "error: %s: %s\n", input, message);
  else fprintf(stderr, "error: %s\n", message);
  exit(EXIT_FAILURE);
}

TensorValue *tensor_alloc(size_t count) {
  if (count > (size_t)-1 / sizeof(TensorValue)) fail(NULL, "tensor allocation size overflow");
  TensorValue *values = malloc(count * sizeof(TensorValue));
  if (!values) fail(NULL, "could not allocate tensor memory");
  return values;
}

static void whitespace(const char **p) {
  while (**p && strchr(" \t\r\n", **p)) ++*p;
}

static void expect(const char **p, char token, const char *input,
                   const char *message) {
  whitespace(p);
  if (**p != token) fail(input, message);
  ++*p;
}

static int digit(char c) { return c >= '0' && c <= '9'; }

static TensorValue number(const char **p, const char *input) {
  whitespace(p);
  const char *start = *p;
  if (**p == '-') ++*p;
  if (**p == '0') ++*p;
  else {
    if (**p < '1' || **p > '9') fail(input, "expected a finite JSON number");
    while (digit(**p)) ++*p;
  }
  if (**p == '.') {
    ++*p;
    if (!digit(**p)) fail(input, "expected digits after decimal point");
    while (digit(**p)) ++*p;
  }
  if (**p == 'e' || **p == 'E') {
    ++*p;
    if (**p == '+' || **p == '-') ++*p;
    if (!digit(**p)) fail(input, "expected digits in exponent");
    while (digit(**p)) ++*p;
  }
  char *end;
  TensorValue value = TENSOR_PARSE(start, &end);
  if (end != *p || !isfinite(value)) fail(input, "expected a finite number");
  return value;
}

static void array(const char **p, TensorInput *input, int dimension,
                  size_t *offset) {
  expect(p, '[', input->name, "expected an array matching the declared shape");
  for (size_t i = 0; i < input->shape[dimension]; ++i) {
    if (i) expect(p, ',', input->name, "array does not match the declared shape");
    if (dimension + 1 < input->rank) array(p, input, dimension + 1, offset);
    else input->values[(*offset)++] = number(p, input->name);
  }
  expect(p, ']', input->name, "array does not match the declared shape");
}

/* DSL names are ASCII identifiers. Decode JSON escapes before matching them. */
static void key(const char **p, char *name, size_t capacity) {
  expect(p, '"', NULL, "expected an input name in double quotes");
  size_t length = 0;
  while (**p != '"') {
    unsigned char c = (unsigned char)**p;
    if (c < 32) fail(NULL, "invalid JSON input name");
    ++*p;
    if (c == '\\') {
      c = (unsigned char)**p;
      if (c == 'u') {
        ++*p;
        unsigned int code = 0;
        for (int i = 0; i < 4; ++i) {
          char h = **p;
          unsigned int hex;
          if (h >= '0' && h <= '9') hex = (unsigned int)(h - '0');
          else if (h >= 'a' && h <= 'f') hex = (unsigned int)(h - 'a' + 10);
          else if (h >= 'A' && h <= 'F') hex = (unsigned int)(h - 'A' + 10);
          else fail(NULL, "invalid Unicode escape in input name");
          code = code * 16 + hex;
          ++*p;
        }
        if (code < 32 || code > 126) fail(NULL, "unknown input name");
        c = (unsigned char)code;
      } else {
        if (!c || !strchr("\"\\/bfnrt", c)) fail(NULL, "invalid JSON escape");
        if (strchr("bfnrt", c)) fail(NULL, "unknown input name");
        ++*p;
      }
    }
    if (length + 1 >= capacity) fail(NULL, "unknown input name");
    name[length++] = (char)c;
  }
  ++*p;
  name[length] = '\0';
}

static void load_inputs(const char *path, TensorInput *inputs, size_t count,
                        char *name, size_t name_capacity) {
  FILE *file = fopen(path, "rb");
  if (!file) { perror(path); exit(EXIT_FAILURE); }
  size_t length = 0, capacity = 4096;
  char *text = malloc(capacity);
  if (!text) fail(NULL, "out of memory reading input file");
  int c;
  while ((c = fgetc(file)) != EOF) {
    if (c == 0) fail(NULL, "NUL byte in JSON input");
    if (length == capacity - 1) {
      if (capacity > (size_t)-1 / 2) fail(NULL, "input file is too large");
      capacity *= 2;
      char *grown = realloc(text, capacity);
      if (!grown) fail(NULL, "out of memory reading input file");
      text = grown;
    }
    text[length++] = (char)c;
  }
  if (ferror(file)) fail(NULL, "could not read input file");
  fclose(file);
  text[length] = '\0';
  const char *p = text;
  expect(&p, '{', NULL, "input JSON must be an object mapping names to arrays");
  whitespace(&p);
  if (*p != '}') {
    for (;;) {
      key(&p, name, name_capacity);
      TensorInput *input = NULL;
      for (size_t i = 0; i < count; ++i)
        if (strcmp(inputs[i].name, name) == 0) { input = &inputs[i]; break; }
      if (!input) fail(name, "unknown input");
      if (input->seen) fail(name, "duplicate input");
      expect(&p, ':', name, "expected ':' after input name");
      size_t offset = 0;
      array(&p, input, 0, &offset);
      input->seen = 1;
      whitespace(&p);
      if (*p != ',') break;
      ++p;
    }
  }
  expect(&p, '}', NULL, "expected end of input object");
  whitespace(&p);
  if (*p) fail(NULL, "unexpected text after input object");
  for (size_t i = 0; i < count; ++i)
    if (!inputs[i].seen) fail(inputs[i].name, "missing input");
  free(text);
}
