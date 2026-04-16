package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"github.com/mitchellh/mapstructure"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/spf13/viper"

	"github.com/bez/llama-ollama-wrapper/internal/flags"
	"github.com/bez/llama-ollama-wrapper/internal/proxy"
	"github.com/bez/llama-ollama-wrapper/internal/runner"
	"github.com/bez/llama-ollama-wrapper/internal/scanner"
	"github.com/bez/llama-ollama-wrapper/internal/selector"
)

func main() {
	// Configure human-readable console logging
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	if err := run(); err != nil {
		log.Fatal().Err(err).Msg("fatal error")
	}
}

const (
	runModeWrapper   = "llama-server-wrapper"
	runModeProxyOnly = "proxy-only"
)

func run() error {
	// ── 1. Locate project root (binary lives in <root>/bin/ or is run via go run) ──
	projectRoot, err := findProjectRoot()
	if err != nil {
		return fmt.Errorf("locating project root: %w", err)
	}

	cfg, err := loadProxyConfig(projectRoot)
	if err != nil {
		return fmt.Errorf("loading proxy config: %w", err)
	}

	mode := cfg.RunMode
	if mode == "" {
		mode = runModeWrapper
	}
	switch mode {
	case runModeWrapper:
		// proceed with wrapper flow below
	case runModeProxyOnly:
		return runProxyOnly()
	default:
		return fmt.Errorf("unknown run mode %q: phải là %q hoặc %q", mode, runModeProxyOnly, runModeWrapper)
	}

	// ── 2. Scan available models ────────────────────────────────────────────────
	models, err := scanner.Scan(projectRoot)
	if err != nil {
		return fmt.Errorf("scanning models: %w", err)
	}

	// ── 3. Interactive model selection ──────────────────────────────────────────
	defaults := map[string]string{}
	if cfg.LlamaServerArgs != nil {
		if v, ok := cfg.LlamaServerArgs[flags.Key(flags.FlagCtxPerSlot)]; ok {
			defaults[flags.Key(flags.FlagCtxPerSlot)] = v
		}
		if v, ok := cfg.LlamaServerArgs[flags.Key(flags.FlagParallel)]; ok {
			defaults[flags.Key(flags.FlagParallel)] = v
		}
	}

	modelPath, overrides, err := selector.Select(models, defaults, os.Stdin, os.Stderr)
	if errors.Is(err, selector.ErrDownloadRequested) {
		fmt.Fprintln(os.Stderr, "Run: make download")
		os.Exit(0)
	}
	if err != nil {
		return fmt.Errorf("selecting model: %w", err)
	}

	if overrides != nil {
		if cfg.LlamaServerArgs == nil {
			cfg.LlamaServerArgs = map[string]string{}
		}
		for k, v := range overrides {
			cfg.LlamaServerArgs[k] = v
		}
	}

	// Find model size for Ollama API responses
	var modelSize int64
	for _, m := range models {
		if m.Path == modelPath {
			modelSize = m.Size
			break
		}
	}

	// ── 4. Start llama-server ───────────────────────────────────────────────────
	fmt.Printf("\nModel   : %s\n", filepath.Base(modelPath))

	// Read host/port of the llama-server from llama_server_args (breaking change)
	var llamaHost, llamaPort string
	if cfg.LlamaServerArgs != nil {
		if v, ok := cfg.LlamaServerArgs[flags.Key(flags.FlagHost)]; ok {
			llamaHost = v
		}
		if v, ok := cfg.LlamaServerArgs[flags.Key(flags.FlagPort)]; ok {
			llamaPort = v
		}
	}
	if llamaHost == "" || llamaPort == "" {
		return fmt.Errorf("llama_server_args must include '%s' and '%s' when running in wrapper mode", flags.Key(flags.FlagHost), flags.Key(flags.FlagPort))
	}

	fmt.Printf("Server  : http://%s:%s\n", llamaHost, llamaPort)
	fmt.Printf("Proxy   : http://%s:%s  (Ollama v%s)\n\n",
		cfg.Host, cfg.Port, cfg.OllamaVersion)

	// Use signal-aware context for graceful shutdown
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		if err := ctx.Err(); err != nil {
			log.Info().Err(err).Msg("shutdown requested")
		} else {
			log.Info().Msg("shutdown requested")
		}
	}()

	log.Info().Str("model", filepath.Base(modelPath)).Msg("starting llama-server")
	r, err := runner.Start(ctx, modelPath, runner.Config{
		Host: llamaHost,
		Port: llamaPort,
		Args: cfg.LlamaServerArgs,
	})
	if err != nil {
		return fmt.Errorf("starting llama-server: %w", err)
	}

	var stopOnce sync.Once
	stopRunner := func() {
		stopOnce.Do(func() {
			if r == nil {
				return
			}
			if stopErr := r.Stop(); stopErr != nil {
				log.Error().Err(stopErr).Msg("stopping llama-server")
			}
		})
	}
	defer stopRunner()

	// Ensure we attempt to stop llama-server as soon as a shutdown signal is received.
	go func() {
		<-ctx.Done()
		stopRunner()
	}()

	log.Info().Msg("Waiting for llama-server to become ready...")
	if err := r.WaitReady(ctx); err != nil {
		stopRunner()
		return fmt.Errorf("llama-server never became ready: %w", err)
	}
	log.Info().Msg("llama-server is ready")

	// ── 5. Start Ollama proxy ───────────────────────────────────────────────────
	srv, err := proxy.New(proxy.Config{
		Host:         cfg.Host,
		Port:         cfg.Port,
		ModelName:    r.ModelName(),
		ModelSize:    modelSize,
		UpstreamBase: r.BaseURL(),
	}, r.CtxSize())
	if err != nil {
		return fmt.Errorf("creating proxy server: %w", err)
	}
	proxyAddr := fmt.Sprintf("http://%s:%s", cfg.Host, cfg.Port)
	log.Info().Str("addr", proxyAddr).Msg("starting Ollama proxy")

	// Run proxy (blocks until context is cancelled or error)
	proxyErr := make(chan error, 1)
	go func() {
		proxyErr <- srv.Start(ctx)
	}()

	// Wait for either the proxy to stop or llama-server to exit
	llamaErr := make(chan error, 1)
	go func() {
		llamaErr <- r.Wait()
	}()

	select {
	case err := <-proxyErr:
		if err != nil {
			log.Error().Err(err).Msg("proxy server stopped with error")
		} else {
			log.Info().Msg("proxy server stopped")
		}
		stop()
	case err := <-llamaErr:
		if err != nil {
			log.Error().Err(err).Msg("llama-server exited with error")
		} else {
			log.Info().Msg("llama-server exited")
		}
		stop()
	}

	log.Info().Msg("shutdown complete")
	return nil
}

// runProxyOnly starts the proxy that forwards to an already-running llama-server.
// Configuration is read from config/proxy.yaml and can be overridden via environment variables.
// Compatible environment variables:
//
//	LLAMA_SERVER_URL or LLAMA_UPSTREAM_BASE — base URL of the llama-server
//	LLAMA_MODEL_NAME                        — model name to advertise via the Ollama API
//	LLAMA_MODEL_SIZE                        — model size in bytes, default 0
//	LLAMA_PROXY_HOST                        — proxy bind address, default 127.0.0.1
//	LLAMA_PROXY_PORT                        — proxy bind port, default 11434
func runProxyOnly() error {
	projectRoot, _ := findProjectRoot()
	cfg, err := loadProxyConfig(projectRoot)
	if err != nil {
		return fmt.Errorf("loading proxy config: %w", err)
	}

	serverURL := cfg.UpstreamBase
	if serverURL == "" {
		return fmt.Errorf("either proxy config upstream_base or UPSTREAM_BASE is required when LLAMA_RUN_MODE=%q", runModeProxyOnly)
	}

	modelName := cfg.ModelName
	if modelName == "" {
		modelName = os.Getenv("LLAMA_MODEL_NAME")
	}
	if modelName == "" {
		return fmt.Errorf("either proxy config model_name or LLAMA_MODEL_NAME is required when LLAMA_RUN_MODE=%q", runModeProxyOnly)
	}

	modelSize := cfg.ModelSize
	if raw := os.Getenv("LLAMA_MODEL_SIZE"); raw != "" {
		v, err := strconv.ParseInt(raw, 10, 64)
		if err != nil {
			return fmt.Errorf("LLAMA_MODEL_SIZE phải là số nguyên, nhận được %q: %w", raw, err)
		}
		modelSize = v
	}

	proxyHost := cfg.Host
	if proxyHost == "" {
		proxyHost = proxy.OllamaProxyHost
	}
	proxyPort := cfg.Port
	if proxyPort == "" {
		proxyPort = proxy.OllamaProxyPort
	}

	cfg.ModelName = modelName

	fmt.Printf("\nMode    : %s\n", runModeProxyOnly)
	fmt.Printf("Upstream: %s\n", serverURL)
	fmt.Printf("Model   : %s\n", modelName)
	fmt.Printf("Proxy   : http://%s:%s  (Ollama v%s)\n\n", proxyHost, proxyPort, cfg.OllamaVersion)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		log.Info().Msg("shutdown requested")
	}()

	srv, err := proxy.New(proxy.Config{
		Host:         proxyHost,
		Port:         proxyPort,
		ModelName:    modelName,
		ModelSize:    modelSize,
		UpstreamBase: serverURL,
	}, resolveContextLength(cfg))
	if err != nil {
		return fmt.Errorf("creating proxy server: %w", err)
	}

	log.Info().Str("addr", fmt.Sprintf("http://%s:%s", proxyHost, proxyPort)).Msg("starting Ollama proxy (proxy-only mode)")
	if err := srv.Start(ctx); err != nil {
		return fmt.Errorf("proxy server error: %w", err)
	}

	log.Info().Msg("shutdown complete")
	return nil
}

func loadProxyConfig(projectRoot string) (proxy.Config, error) {
	cfg := proxy.Config{
		Host:          proxy.OllamaProxyHost,
		Port:          proxy.OllamaProxyPort,
		ModelSize:     0,
		OllamaVersion: proxy.OllamaVersion,
	}

	v := viper.NewWithOptions(viper.ExperimentalBindStruct(), viper.ExperimentalFinder())

	v.SetDefault("HOST", cfg.Host)
	v.SetDefault("PORT", cfg.Port)
	v.SetDefault("RUN_MODE", runModeWrapper)
	v.SetDefault("MODEL_SIZE", cfg.ModelSize)
	// Breaking change: host/port for llama-server are now nested under llama_server_args
	v.SetDefault("LLAMA_SERVER_ARGS__"+flags.Key(flags.FlagHost), "127.0.0.1")
	v.SetDefault("LLAMA_SERVER_ARGS__"+flags.Key(flags.FlagPort), "18888")
	v.SetDefault("OLLAMA_VERSION", cfg.OllamaVersion)

	v.SetConfigName("proxy")
	v.SetConfigType("yaml")
	v.AddConfigPath("config/")
	v.AddConfigPath("./config/")
	v.AddConfigPath("../../config/") // Look for config needed for tests.
	v.AddConfigPath(".")
	if projectRoot != "" {
		v.AddConfigPath(filepath.Join(projectRoot, "config"))
	}

	v.SetEnvKeyReplacer(strings.NewReplacer(".", "__"))
	v.AutomaticEnv()

	if err := v.ReadInConfig(); err != nil {
		return cfg, fmt.Errorf("reading proxy config: %w", err)
	}

	if err := v.Unmarshal(&cfg, viper.DecodeHook(
		mapstructure.ComposeDecodeHookFunc(
			mapstructure.StringToTimeDurationHookFunc(),
		),
	)); err != nil {
		return cfg, fmt.Errorf("parsing proxy config: %w", err)
	}

	return cfg, nil
}

func resolveContextLength(cfg proxy.Config) int {
	if cfg.LlamaServerArgs != nil {
		// Prefer explicit ctx-size if provided
		if raw, ok := cfg.LlamaServerArgs[flags.Key(flags.FlagCtxSize)]; ok {
			if v, err := strconv.Atoi(raw); err == nil && v > 0 {
				return v
			}
		}
		// Fallback to ctx-per-slot (not multiplied by parallel here)
		if raw, ok := cfg.LlamaServerArgs[flags.Key(flags.FlagCtxPerSlot)]; ok {
			if v, err := strconv.Atoi(raw); err == nil && v > 0 {
				return v
			}
		}
	}
	return 0
}

func findProjectRoot() (string, error) {
	// Try executable path first
	exe, err := os.Executable()
	if err == nil {
		dir := filepath.Dir(exe)
		// go run puts the binary under /tmp/go-build... on all platforms
		if !isTempPath(dir) {
			if root, ok := findGoMod(dir); ok {
				return root, nil
			}
		}
	}

	// Fallback: walk up from cwd
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	if root, ok := findGoMod(cwd); ok {
		return root, nil
	}

	// Last resort: use cwd as-is
	return cwd, nil
}

func findGoMod(start string) (string, bool) {
	dir := start
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", false
}

func isTempPath(path string) bool {
	tmp := os.TempDir()
	if strings.HasPrefix(path, tmp) {
		return true
	}
	// macOS specific
	if runtime.GOOS == "darwin" && strings.HasPrefix(path, "/var/") {
		return true
	}
	return false
}
