package runner

import (
	"strings"
	"testing"

	"github.com/bez/llama-ollama-wrapper/internal/flags"
)

// parseArgs converts an args slice into a map of key->value. Flags without
// values are represented with value "true". Flags with value "false" are
// expected to be omitted by buildArgs.
func parseArgs(args []string) map[string]string {
    m := make(map[string]string)
    for i := 0; i < len(args); i++ {
        a := args[i]
        if !strings.HasPrefix(a, "--") {
            continue
        }
        key := strings.TrimPrefix(a, "--")
        if i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
            m[key] = args[i+1]
            i++
        } else {
            m[key] = "true"
        }
    }
    return m
}

func TestBuildArgs_OverridesAndBooleans(t *testing.T) {
    modelPath := "/path/to/model.gguf"
    modelName := "model"
    host := "127.0.0.1"
    port := "18888"
    overrides := map[string]string{
        flags.Key(flags.FlagCtxPerSlot):      "1000",
        flags.Key(flags.FlagNgpuLayers):      "10",
        flags.Key(flags.FlagEmbedding):       "false",
        "custom-flag":                       "true",
        flags.Key(flags.FlagModel):           "bad-model",
        flags.Key(flags.FlagAlias):           "bad-alias",
        flags.Key(flags.FlagHost):            "9.9.9.9",
        flags.Key(flags.FlagPort):            "9999",
    }

    args, _ := buildArgs(modelPath, modelName, host, port, overrides)
    m := parseArgs(args)

    if got := m[flags.Key(flags.FlagCtxSize)]; got != "1000" {
        t.Fatalf("ctx-size = %q; want %q", got, "1000")
    }
    if _, ok := m[flags.Key(flags.FlagCtxPerSlot)]; ok {
        t.Fatalf("ctx-per-slot present; want omitted when used to calculate ctx-size")
    }
    if got := m[flags.Key(flags.FlagNgpuLayers)]; got != "10" {
        t.Fatalf("n-gpu-layers = %q; want %q", got, "10")
    }
    if _, ok := m[flags.Key(flags.FlagEmbedding)]; ok {
        t.Fatalf("embedding flag present; want omitted when false")
    }
    if got := m["custom-flag"]; got != "true" {
        t.Fatalf("custom-flag = %q; want %q", got, "true")
    }
    // model, alias, host and port must not be overridden by overrides map
    if got := m[flags.Key(flags.FlagModel)]; got != modelPath {
        t.Fatalf("model = %q; want %q", got, modelPath)
    }
    if got := m[flags.Key(flags.FlagAlias)]; got != modelName {
        t.Fatalf("alias = %q; want %q", got, modelName)
    }
    if got := m[flags.Key(flags.FlagHost)]; got != host {
        t.Fatalf("host = %q; want %q", got, host)
    }
    if got := m[flags.Key(flags.FlagPort)]; got != port {
        t.Fatalf("port = %q; want %q", got, port)
    }
}

func TestBuildArgs_ExtraOrdering(t *testing.T) {
    modelPath := "/path/m.gguf"
    modelName := "m"
    host := "h"
    port := "p"
    overrides := map[string]string{
        "z-arg": "1",
        "a-arg": "2",
    }

    args, _ := buildArgs(modelPath, modelName, host, port, overrides)

    idx := func(key string) int {
        for i, a := range args {
            if a == "--"+key {
                return i
            }
        }
        return -1
    }
    ai := idx("a-arg")
    zi := idx("z-arg")
    if ai == -1 || zi == -1 {
        t.Fatalf("missing extra args: a-arg index %d z-arg index %d", ai, zi)
    }
    if ai > zi {
        t.Fatalf("extra args ordering wrong: a-arg idx %d > z-arg idx %d", ai, zi)
    }
}
