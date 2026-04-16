// Package runner manages the llama-server subprocess.
// It starts the server with the proper flags and polls /health until ready.
package runner

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/bez/llama-ollama-wrapper/internal/flags"
	"github.com/rs/zerolog/log"
)

const (
	defaultLlamaServerHost = "127.0.0.1"
	defaultLlamaServerPort = "18888"
)

type Config struct {
	Host string
	Port string
	// Args is a key-value map of llama-server CLI flags.
	// Boolean flags can be enabled with "true" and disabled with "false".
	Args map[string]string
}

// Runner holds a running llama-server process.
type Runner struct {
	cmd       *exec.Cmd
	modelName string
	baseURL   string
	waitErr   error
	waitDone  chan struct{}
	ctxSize   int
}

// Start launches llama-server with the given GGUF model file.
// The process inherits stdin/stdout/stderr so the user sees live output.
// The caller must call Wait() or Stop() to clean up the process.
func Start(ctx context.Context, modelPath string, cfg Config) (*Runner, error) {
	if _, err := exec.LookPath("llama-server"); err != nil {
		return nil, fmt.Errorf("llama-server not found in PATH: %w (install with: brew install llama.cpp)", err)
	}

	modelName := strings.TrimSuffix(filepath.Base(modelPath), ".gguf")

	host := cfg.Host
	if host == "" {
		host = defaultLlamaServerHost
	}
	port := cfg.Port
	if port == "" {
		port = defaultLlamaServerPort
	}
	baseURL := fmt.Sprintf("http://%s:%s", host, port)

	args, ctxSize := buildArgs(modelPath, modelName, host, port, cfg.Args)

	cmd := exec.Command("llama-server", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = nil
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting llama-server: %w", err)
	}

	runner := &Runner{cmd: cmd, modelName: modelName, baseURL: baseURL, waitDone: make(chan struct{}), ctxSize: ctxSize}
	go func() {
		runner.waitErr = cmd.Wait()
		close(runner.waitDone)
	}()

	return runner, nil
}

func buildArgs(modelPath, modelName, host, port string, overrides map[string]string) ([]string, int) {
	defaults := []struct {
		key   string
		value string
	}{
		{flags.Key(flags.FlagModel), modelPath},
		{flags.Key(flags.FlagAlias), modelName},
		{flags.Key(flags.FlagNgpuLayers), "99"},
		{flags.Key(flags.FlagCtxSize), "262144"},
		{flags.Key(flags.FlagParallel), "1"},
		{flags.Key(flags.FlagCacheTypeK), "q8_0"},
		{flags.Key(flags.FlagCacheTypeV), "q8_0"},
		{flags.Key(flags.FlagFlashAttn), "auto"},
		{flags.Key(flags.FlagEmbedding), "true"},
		{flags.Key(flags.FlagHost), host},
		{flags.Key(flags.FlagPort), port},
	}
	finalCtxSize := 262144

	storage := make(map[string]string, len(defaults)+len(overrides))
	for _, item := range defaults {
		storage[item.key] = item.value
	}
	for k, v := range overrides {
		switch k {
		case flags.Key(flags.FlagModel), flags.Key(flags.FlagAlias), flags.Key(flags.FlagHost), flags.Key(flags.FlagPort):
			continue
		default:
			storage[k] = v
		}
	}

	arguments := make([]string, 0, len(storage)*2)
	for _, item := range defaults {
		appendArg(&arguments, item.key, storage[item.key])
	}

	// Add any extra custom args not present in defaults in deterministic order.
	extraKeys := make([]string, 0, len(storage))
	for k := range storage {
		if !containsKey(defaults, k) && k != flags.Key(flags.FlagCtxPerSlot) {
			extraKeys = append(extraKeys, k)
		}
	}
	// handle calculation of the ctx-size parameter
	if ctxPerSlot, ok := overrides[flags.Key(flags.FlagCtxPerSlot)]; ok {
		if parallel, ok := storage[flags.Key(flags.FlagParallel)]; ok {
			ctxPerSlotInt, err1 := strconv.Atoi(ctxPerSlot)
			parallelInt, err2 := strconv.Atoi(parallel)
			if err1 == nil && err2 == nil {
				finalCtxSize = ctxPerSlotInt
				storage[flags.Key(flags.FlagCtxSize)] = fmt.Sprintf("%d", ctxPerSlotInt*parallelInt)
				i := slices.Index(arguments, "--"+flags.Key(flags.FlagCtxSize))
				if i >= 0 && i+1 < len(arguments) {
					arguments[i+1] = storage[flags.Key(flags.FlagCtxSize)]
				}
				log.Info().Msgf("calculated ctx-size=%s from ctx-per-slot=%s and parallel=%s", storage[flags.Key(flags.FlagCtxSize)], ctxPerSlot, parallel)
				delete(storage, flags.Key(flags.FlagCtxPerSlot))
				// Remove ctx-per-slot from extraKeys if it exists
			}
		}
	}

	// After all modifications, parse final ctx-size to return as int.
	if totalCtxSize, ok1 := storage[flags.Key(flags.FlagCtxSize)]; ok1 {
		if parallelSlot, ok2 := storage[flags.Key(flags.FlagParallel)]; ok2 {
			if totalCtxSizeInt, err1 := strconv.Atoi(totalCtxSize); err1 == nil {
				if parallelSlotInt, err2 := strconv.Atoi(parallelSlot); err2 == nil && parallelSlotInt > 0 {
					finalCtxSize = totalCtxSizeInt / parallelSlotInt
				}
			}
		}
	}

	sort.Strings(extraKeys)
	for _, k := range extraKeys {
		appendArg(&arguments, k, storage[k])
	}

	return arguments, finalCtxSize
}

func containsKey(defaults []struct{ key, value string }, key string) bool {
	for _, item := range defaults {
		if item.key == key {
			return true
		}
	}
	return false
}

func appendArg(arguments *[]string, key, value string) {
	if strings.EqualFold(value, "false") {
		return
	}
	*arguments = append(*arguments, "--"+key)
	if value != "" && !strings.EqualFold(value, "true") {
		*arguments = append(*arguments, value)
	}
}

// ModelName returns the alias passed to llama-server (filename without .gguf).
func (r *Runner) ModelName() string {
	return r.modelName
}

// BaseURL returns the llama-server base URL.
func (r *Runner) BaseURL() string {
	if r == nil || r.baseURL == "" {
		return fmt.Sprintf("http://%s:%s", defaultLlamaServerHost, defaultLlamaServerPort)
	}
	return r.baseURL
}

// CtxSize returns the final ctx-size used to start llama-server.
// If unknown, returns the default 262144.
func (r *Runner) CtxSize() int {
	if r == nil || r.ctxSize == 0 {
		return 262144
	}
	return r.ctxSize
}

// Stop terminates the llama-server process if it is still running.
// It attempts a graceful shutdown first and falls back to kill.
func (r *Runner) Stop() error {
	if r == nil || r.cmd == nil || r.cmd.Process == nil {
		return nil
	}

	if r.cmd.ProcessState != nil && r.cmd.ProcessState.Exited() {
		log.Info().Msg("llama-server already exited before stop")
		<-r.waitDone
		return r.waitErr
	}

	log.Info().Msg("stopping llama-server")
	var err error
	if runtime.GOOS == "windows" {
		err = r.cmd.Process.Kill()
	} else {
		err = syscall.Kill(-r.cmd.Process.Pid, syscall.SIGTERM)
	}
	if err != nil && !errors.Is(err, os.ErrProcessDone) {
		return err
	}

	select {
	case <-r.waitDone:
		log.Info().Msg("llama-server stopped gracefully")
		return r.waitErr
	case <-time.After(5 * time.Second):
		log.Warn().Msg("llama-server did not exit after SIGTERM, killing process group")
		if runtime.GOOS == "windows" {
			if killErr := r.cmd.Process.Kill(); killErr != nil && !errors.Is(killErr, os.ErrProcessDone) {
				return killErr
			}
		} else {
			if killErr := syscall.Kill(-r.cmd.Process.Pid, syscall.SIGKILL); killErr != nil && !errors.Is(killErr, os.ErrProcessDone) {
				return killErr
			}
		}
		<-r.waitDone
		return r.waitErr
	}
}

// WaitReady polls GET /health until it returns HTTP 200 or the context is cancelled.
// It uses a 5-minute timeout to handle large models being loaded.
func (r *Runner) WaitReady(ctx context.Context) error {
	deadline := time.Now().Add(5 * time.Minute)
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	client := &http.Client{Timeout: 2 * time.Second}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case t := <-ticker.C:
			if t.After(deadline) {
				return fmt.Errorf("llama-server failed to become ready within 5 minutes")
			}
			resp, err := client.Get(r.baseURL + "/health")
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					return nil
				}
			}
			// Process may have already exited — check without blocking
			if r.cmd.ProcessState != nil && r.cmd.ProcessState.Exited() {
				return fmt.Errorf("llama-server exited unexpectedly (code %d)", r.cmd.ProcessState.ExitCode())
			}
		}
	}
}

// Wait blocks until the llama-server process exits.
func (r *Runner) Wait() error {
	if r == nil {
		return nil
	}
	<-r.waitDone
	return r.waitErr
}
