package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"

	"github.com/mitchellh/mapstructure"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/spf13/viper"

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
		// tiếp tục flow llama-server-wrapper bên dưới
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
	modelPath, err := selector.Select(models)
	if err != nil {
		return fmt.Errorf("selecting model: %w", err)
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
	fmt.Printf("Server  : http://%s:%s\n", cfg.LlamaServerHost, cfg.LlamaServerPort)
	fmt.Printf("Proxy   : http://%s:%s  (Ollama v%s)\n\n",
		cfg.Host, cfg.Port, cfg.OllamaVersion)

	// Use signal-aware context for graceful shutdown
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM, os.Kill)
	defer stop()
	go func() {
		<-ctx.Done()
		if ctx.Err() != nil {
			log.Info().Err(ctx.Err()).Msg("shutdown requested")
		} else {
			log.Info().Msg("shutdown requested")
		}
	}()

	log.Info().Str("model", filepath.Base(modelPath)).Msg("starting llama-server")
	r, err := runner.Start(ctx, modelPath, runner.Config{
		Host: cfg.LlamaServerHost,
		Port: cfg.LlamaServerPort,
		Args: cfg.LlamaServerArgs,
	})
	if err != nil {
		return fmt.Errorf("starting llama-server: %w", err)
	}
	defer func() {
		if stopErr := r.Stop(); stopErr != nil {
			log.Error().Err(stopErr).Msg("stopping llama-server")
		}
	}()

	// Ensure we attempt to stop llama-server as soon as a shutdown signal is received.
	go func() {
		<-ctx.Done()
		if r != nil {
			if stopErr := r.Stop(); stopErr != nil {
				log.Error().Err(stopErr).Msg("stopping llama-server on shutdown signal")
			}
		}
	}()

	log.Info().Msg("Waiting for llama-server to become ready...")
	if err := r.WaitReady(ctx); err != nil {
		if stopErr := r.Stop(); stopErr != nil {
			log.Error().Err(stopErr).Msg("stopping llama-server after failed readiness")
		}
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
	})
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

// runProxyOnly khởi động proxy kết nối thẳng vào llama-server đang chạy sẵn.
// Cấu hình được đọc từ config/proxy.yaml và có thể ghi đè bằng environment variables.
// Các biến env tương thích:
//
//	LLAMA_SERVER_URL hoặc LLAMA_UPSTREAM_BASE  — base URL của llama-server
//	LLAMA_MODEL_NAME                         — tên model quảng bá qua Ollama API
//	LLAMA_MODEL_SIZE                         — kích thước model tính bằng bytes, mặc định 0
//	LLAMA_PROXY_HOST                         — địa chỉ bind của proxy, mặc định 127.0.0.1
//	LLAMA_PROXY_PORT                         — port của proxy, mặc định 11434
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

	cfg.UpstreamBase = cfg.UpstreamBase
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
	})
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

// findProjectRoot walks up from the executable's directory to find the
// project root (the directory containing go.mod).
// When running via `go run` the executable is in a temp dir, so we fall back
// to the current working directory.
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
	v.SetDefault("LLAMA_SERVER_HOST", "127.0.0.1")
	v.SetDefault("LLAMA_SERVER_PORT", "18888")
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
	)); err != nil { // Handle errors reading the config file
		panic(fmt.Errorf("fatal error config file: %s", err))
	}

	return cfg, nil
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
	if len(path) >= len(tmp) && path[:len(tmp)] == tmp {
		return true
	}
	// macOS specific
	if runtime.GOOS == "darwin" && len(path) > 4 && path[:5] == "/var/" {
		return true
	}
	return false
}
