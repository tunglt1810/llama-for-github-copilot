package selector

import (
	"bytes"
	"strings"
	"testing"

	"github.com/bez/llama-ollama-wrapper/internal/scanner"
)

func TestReadChoice_Valid(t *testing.T) {
	in := strings.NewReader("2\n")
	out := &bytes.Buffer{}
	n, err := readChoice(3, in, out)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n != 2 {
		t.Fatalf("expected 2, got %d", n)
	}
}

func TestReadChoice_Invalid(t *testing.T) {
	in := strings.NewReader("x\n")
	out := &bytes.Buffer{}
	if _, err := readChoice(3, in, out); err == nil {
		t.Fatalf("expected error for invalid input")
	}
}

func TestReadIntWithDefault_UsesDefaultWhenBlank(t *testing.T) {
	in := strings.NewReader("\n")
	out := &bytes.Buffer{}
	v, err := readIntWithDefault("ctx-per-slot", "128", in, out)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v != "128" {
		t.Fatalf("expected default 128, got %q", v)
	}
}

func TestReadIntWithDefault_ValidThenInvalid(t *testing.T) {
	in := strings.NewReader("foo\n256\n")
	out := &bytes.Buffer{}
	v, err := readIntWithDefault("ctx-per-slot", "128", in, out)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v != "256" {
		t.Fatalf("expected 256, got %q", v)
	}
}

func TestSelect_SingleModel_NonTTY(t *testing.T) {
	models := []scanner.Model{{Path: "/tmp/m1.gguf", Size: 123}}
	in := strings.NewReader("")
	out := &bytes.Buffer{}
	path, overrides, err := Select(models, nil, in, out)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if path != models[0].Path {
		t.Fatalf("expected %s, got %s", models[0].Path, path)
	}
	if overrides != nil {
		t.Fatalf("expected nil overrides for non-TTY, got %v", overrides)
	}
}

func TestSelect_MultiModelChoice_NonTTY(t *testing.T) {
	models := []scanner.Model{{Path: "/tmp/m1.gguf", Size: 1}, {Path: "/tmp/m2.gguf", Size: 2}}
	in := strings.NewReader("2\n")
	out := &bytes.Buffer{}
	path, overrides, err := Select(models, nil, in, out)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if path != models[1].Path {
		t.Fatalf("expected %s, got %s", models[1].Path, path)
	}
	if overrides != nil {
		t.Fatalf("expected nil overrides for non-TTY, got %v", overrides)
	}
}

func TestModelLabel_WithRepoPrefix(t *testing.T) {
	path := "/home/user/.cache/huggingface/hub/models--repo-name/snapshots/abcdef/qwen3.5-9B.q8_0.gguf"
	got := modelLabel(path)
	want := "repo-name:q8_0"
	if got != want {
		t.Fatalf("modelLabel mismatch: want %q got %q", want, got)
	}
}
