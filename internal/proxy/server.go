package proxy

import (
	"context"
	"fmt"
	"time"

	"github.com/labstack/echo/v5"
	"github.com/labstack/echo/v5/middleware"
	"github.com/rs/zerolog/log"
)

const (
	// OllamaProxyHost binds to localhost only to avoid unintended network exposure.
	// SECURITY: never bind to 0.0.0.0 unless the user explicitly requests it.
	OllamaProxyHost = "127.0.0.1"
	OllamaProxyPort = "11434"
)

// Config holds the configuration for the Ollama proxy server.
type Config struct {
	// Host is the bind address for the proxy. Default: OllamaProxyHost ("127.0.0.1").
	Host string `mapstructure:"host"`
	// Port is the bind port for the proxy. Default: OllamaProxyPort ("11434").
	Port string `mapstructure:"port"`
	// ModelName is the model name advertised via the Ollama API.
	ModelName string `mapstructure:"model_name"`
	// ModelSize is the model file size in bytes (used for /api/tags, /api/ps).
	ModelSize int64 `mapstructure:"model_size"`
	// UpstreamBase is the base URL of the llama-server (e.g. "http://127.0.0.1:18888").
	UpstreamBase string `mapstructure:"upstream_base"`
	// LlamaServerArgs contains options to run llama-server in key-value format.
	// Example:
	// llama_server_args:
	//   n-gpu-layers: "99"
	//   ctx-per-slot: "262144"
	//   flash-attn: "auto"
	LlamaServerArgs map[string]string `mapstructure:"llama_server_args"`
	// RunMode allows reading the run mode value from the proxy.yaml configuration or from LLAMA_RUN_MODE.
	// Note: the actual `ctx-size` value is provided by the Runner when the proxy runs as a wrapper.
	RunMode string `mapstructure:"run_mode"`
	// OllamaVersion is the Ollama version reported by the proxy via /api/version.
	OllamaVersion string `mapstructure:"ollama_version"`
	// Note: The llama-server host/port information is now placed inside
	// `LlamaServerArgs` with keys `host` and `port`.
	// Example YAML:
	// llama_server_args:
	//   host: 127.0.0.1
	//   port: 50000
}

// Server is an Ollama-compatible HTTP proxy server.
type Server struct {
	echo *echo.Echo
	cfg  Config
}

// New creates a new Ollama proxy Server from cfg.
// cfg.Host and cfg.Port will use OllamaProxyHost/OllamaProxyPort as defaults if empty.
func New(cfg Config, contextSize int) (*Server, error) {
	if cfg.Host == "" {
		cfg.Host = OllamaProxyHost
	}
	if cfg.Port == "" {
		cfg.Port = OllamaProxyPort
	}
	if cfg.OllamaVersion == "" {
		cfg.OllamaVersion = OllamaVersion
	}

	h, err := newHandlers(cfg.ModelName, cfg.ModelSize, cfg.UpstreamBase, cfg.OllamaVersion, contextSize)
	if err != nil {
		return nil, err
	}

	e := echo.New()

	// Middleware
	e.Use(middleware.Recover())
	e.Use(middleware.RequestLoggerWithConfig(middleware.RequestLoggerConfig{
		LogMethod:  true,
		LogURI:     true,
		LogStatus:  true,
		LogLatency: true,
		// LogRemoteIP: true,
		LogValuesFunc: func(c *echo.Context, v middleware.RequestLoggerValues) error {
			log.Info().
				Str("method", v.Method).
				Str("uri", v.URI).
				Int("status", v.Status).
				Dur("latency", v.Latency).
				Str("remote_ip", v.RemoteIP).
				Msg("request")
			return nil
		},
	}))

	// ── Ollama native API ────────────────────────────────────────────────────
	e.GET("/api/version", h.version)
	e.GET("/api/tags", h.tags)
	e.GET("/api/ps", h.ps)
	e.POST("/api/show", h.show)
	e.POST("/api/chat", h.chat)
	e.POST("/api/generate", h.generate)
	e.POST("/api/embed", h.embed)

	// ── OpenAI-compatible pass-through (/v1/*) ──────────────────────────────
	// Allows clients that already speak OpenAI format to use port 11434 directly.
	e.GET("/v1/models", h.v1Models)
	e.Any("/v1/*", h.passthrough)

	return &Server{echo: e, cfg: cfg}, nil
}

// Start runs the proxy server on cfg.Host:cfg.Port.
// Blocks until ctx is cancelled, then performs a graceful shutdown (timeout 10s).
func (s *Server) Start(ctx context.Context) error {
	addr := fmt.Sprintf("%s:%s", s.cfg.Host, s.cfg.Port)
	log.Info().
		Str("addr", "http://"+addr).
		Str("ollama_version", s.cfg.OllamaVersion).
		Msg("Ollama proxy listening")

	sc := echo.StartConfig{
		Address:         addr,
		HideBanner:      true,
		HidePort:        true,
		GracefulTimeout: 10 * time.Second,
	}

	// sc.Start blocks until ctx is cancelled, then performs graceful shutdown.
	return sc.Start(ctx, s.echo)
}
