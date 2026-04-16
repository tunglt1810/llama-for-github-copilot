// Package selector provides an interactive CLI model selection menu.
// It replicates the select_model() UX from the original start-server.sh.
package selector

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/mattn/go-isatty"

	"github.com/bez/llama-ollama-wrapper/internal/flags"
	"github.com/bez/llama-ollama-wrapper/internal/scanner"
)

var (
	ErrDownloadRequested = errors.New("download requested")
	ErrNoModels          = errors.New("no models found")
)

const downloadHint = "Run: make download"
const defaultDownloadDesc = "Qwen3.5-9B Q8_0, ~9.5 GB"

// Select presents an interactive numbered list of models to `out` and reads
// the user's choice from `in`. It returns the path of the selected model and
// a map of overrides for llama-server args (e.g. ctx-per-slot, parallel).
// Defaults for prompts are taken from `defaults` (map keyed by flags.Key(...)).
func Select(models []scanner.Model, defaults map[string]string, in io.Reader, out io.Writer) (string, map[string]string, error) {
	if in == nil {
		in = os.Stdin
	}
	if out == nil {
		out = os.Stderr
	}

	if len(models) == 0 {
		return "", nil, promptDownload(in, out)
	}

	var modelPath string
	if len(models) == 1 {
		fmt.Fprintf(out, "Using model: %s\n", modelLabel(models[0].Path))
		modelPath = models[0].Path
	} else {
		printMenu(models, out)
		choice, err := readChoice(len(models), in, out)
		if err != nil {
			return "", nil, err
		}
		if choice == 0 {
			fmt.Fprintln(out, downloadHint)
			return "", nil, ErrDownloadRequested
		}
		modelPath = models[choice-1].Path
	}

	// If input is not a TTY, skip additional prompts and return defaults as-is.
	if !isTTY(in) {
		return modelPath, nil, nil
	}

	// Prompt for ctx-per-slot and parallel, using defaults provided.
	overrides := make(map[string]string)
	ctxKey := flags.Key(flags.FlagCtxPerSlot)
	parKey := flags.Key(flags.FlagParallel)

	defaultCtx := ""
	defaultPar := ""
	if defaults != nil {
		defaultCtx = defaults[ctxKey]
		defaultPar = defaults[parKey]
	}

	ctxVal, err := readIntWithDefault(string(flags.FlagCtxPerSlot), defaultCtx, in, out)
	if err != nil {
		return "", nil, err
	}
	if ctxVal != "" && ctxVal != defaultCtx {
		overrides[ctxKey] = ctxVal
	}

	parVal, err := readIntWithDefault(string(flags.FlagParallel), defaultPar, in, out)
	if err != nil {
		return "", nil, err
	}
	if parVal != "" && parVal != defaultPar {
		overrides[parKey] = parVal
	}

	return modelPath, overrides, nil
}

func printMenu(models []scanner.Model, out io.Writer) {
	fmt.Fprintln(out, "📦 Available models:")
	fmt.Fprintln(out, "")
	for i, m := range models {
		fmt.Fprintf(out, "  [%d] %s (%s)\n", i+1, modelLabel(m.Path), humanSize(m.Size))
	}
	fmt.Fprintf(out, "  [0] Download new model (%s)\n", defaultDownloadDesc)
	fmt.Fprintln(out, "")
}

func modelLabel(path string) string {
	name := strings.TrimSuffix(filepath.Base(path), ".gguf")
	label := name

	parts := strings.Split(filepath.ToSlash(path), "/")
	for _, part := range parts {
		if strings.HasPrefix(part, "models--") {
			repoName := strings.TrimPrefix(part, "models--")
			if i := strings.LastIndex(name, "."); i > 0 {
				label = repoName + ":" + name[i+1:]
			} else {
				label = repoName
			}
			break
		}
	}

	return label
}

func readChoice(max int, in io.Reader, out io.Writer) (int, error) {
	fmt.Fprintf(out, "Select model (0-%d): ", max)
	reader := bufio.NewReader(in)
	line, err := reader.ReadString('\n')
	if err != nil {
		return 0, fmt.Errorf("reading input: %w", err)
	}
	raw := strings.TrimSpace(line)
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 || n > max {
		return 0, fmt.Errorf("invalid selection: %q", raw)
	}
	return n, nil
}

func promptDownload(in io.Reader, out io.Writer) error {
	fmt.Fprintln(out, "❌ No models found locally.")
	fmt.Fprintln(out, "")
	fmt.Fprintf(out, "Would you like to download %s?\n", defaultDownloadDesc)
	fmt.Fprint(out, "Download? (y/n): ")

	if !isTTY(in) {
		fmt.Fprintln(out, downloadHint)
		return ErrNoModels
	}

	reader := bufio.NewReader(in)
	line, err := reader.ReadString('\n')
	if err != nil {
		return fmt.Errorf("reading input: %w", err)
	}
	answer := strings.TrimSpace(strings.ToLower(line))
	if answer == "y" || answer == "yes" {
		fmt.Fprintln(out, downloadHint)
		return ErrDownloadRequested
	}
	fmt.Fprintln(out, "Run 'make download' to fetch the model, then try again.")
	return ErrNoModels
}

func readIntWithDefault(name string, defaultVal string, in io.Reader, out io.Writer) (string, error) {
	disp := ""
	if defaultVal != "" {
		disp = fmt.Sprintf(" [%s]", defaultVal)
	}
	reader := bufio.NewReader(in)
	for {
		fmt.Fprintf(out, "%s%s: ", name, disp)
		line, err := reader.ReadString('\n')
		if err != nil {
			return "", fmt.Errorf("reading input: %w", err)
		}
		raw := strings.TrimSpace(line)
		if raw == "" {
			return defaultVal, nil
		}
		if _, err := strconv.Atoi(raw); err != nil || strings.HasPrefix(raw, "-") {
			fmt.Fprintln(out, "Invalid selection; enter a positive integer or leave blank for default.")
			continue
		}
		return raw, nil
	}
}

func isTTY(in io.Reader) bool {
	if f, ok := in.(*os.File); ok {
		return isatty.IsTerminal(f.Fd())
	}
	return false
}

func humanSize(bytes int64) string {
	gb := float64(bytes) / (1024 * 1024 * 1024)
	return fmt.Sprintf("%.1f GB", gb)
}
