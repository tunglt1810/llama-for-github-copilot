// Package selector provides an interactive CLI model selection menu.
// It replicates the select_model() UX from the original start-server.sh.
package selector

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/bez/llama-ollama-wrapper/internal/scanner"
)

const downloadHint = "Run: make download"
const defaultDownloadDesc = "Qwen3.5-9B Q8_0, ~9.5 GB"

// Select presents an interactive numbered list of models to stderr and reads
// the user's choice from stdin. It returns the path of the selected model.
// If no models are found it prompts the user to download.
func Select(models []scanner.Model) (string, error) {
	if len(models) == 0 {
		return "", promptDownload()
	}

	if len(models) == 1 {
		fmt.Fprintf(os.Stderr, "Using model: %s\n", modelLabel(models[0].Path))
		return models[0].Path, nil
	}

	printMenu(models)

	choice, err := readChoice(len(models))
	if err != nil {
		return "", err
	}
	if choice == 0 {
		fmt.Fprintln(os.Stderr, downloadHint)
		os.Exit(0)
	}
	return models[choice-1].Path, nil
}

func printMenu(models []scanner.Model) {
	fmt.Fprintln(os.Stderr, "📦 Available models:")
	fmt.Fprintln(os.Stderr, "")
	for i, m := range models {
		fmt.Fprintf(os.Stderr, "  [%d] %s (%s)\n", i+1, modelLabel(m.Path), humanSize(m.Size))
	}
	fmt.Fprintf(os.Stderr, "  [0] Download new model (%s)\n", defaultDownloadDesc)
	fmt.Fprintln(os.Stderr, "")
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

func readChoice(max int) (int, error) {
	fmt.Fprintf(os.Stderr, "Select model (0-%d): ", max)
	reader := bufio.NewReader(os.Stdin)
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

func promptDownload() error {
	fmt.Fprintln(os.Stderr, "❌ No models found locally.")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintf(os.Stderr, "Would you like to download %s?\n", defaultDownloadDesc)
	fmt.Fprint(os.Stderr, "Download? (y/n): ")

	reader := bufio.NewReader(os.Stdin)
	line, err := reader.ReadString('\n')
	if err != nil {
		return fmt.Errorf("reading input: %w", err)
	}
	answer := strings.TrimSpace(strings.ToLower(line))
	if answer == "y" || answer == "yes" {
		fmt.Fprintln(os.Stderr, downloadHint)
		os.Exit(0)
	}
	fmt.Fprintln(os.Stderr, "Run 'make download' to fetch the model, then try again.")
	os.Exit(1)
	return nil
}

func humanSize(bytes int64) string {
	gb := float64(bytes) / (1024 * 1024 * 1024)
	return fmt.Sprintf("%.1f GB", gb)
}
