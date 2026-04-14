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

// Config chứa cấu hình cho Ollama proxy server.
type Config struct {
	// Host là địa chỉ bind của proxy. Mặc định: OllamaProxyHost ("127.0.0.1").
	Host string `mapstructure:"host"`
	// Port là port bind của proxy. Mặc định: OllamaProxyPort ("11434").
	Port string `mapstructure:"port"`
	// ModelName là tên model quảng bá qua Ollama API.
	ModelName string `mapstructure:"model_name"`
	// ModelSize là kích thước file model tính bằng bytes (dùng cho /api/tags, /api/ps).
	ModelSize int64 `mapstructure:"model_size"`
	// UpstreamBase là base URL của llama-server (ví dụ: "http://127.0.0.1:18888").
	UpstreamBase string `mapstructure:"upstream_base"`
	// LlamaServerArgs chứa các tuỳ chọn chạy llama-server theo định dạng key-value.
	// Ví dụ:
	// llama_server_args:
	//   n-gpu-layers: "99"
	//   ctx-size: "262144"
	//   flash-attn: "auto"
	LlamaServerArgs map[string]string `mapstructure:"llama_server_args"`
	// RunMode cho phép đọc giá trị run mode từ cấu hình proxy.yaml hoặc LLAMA_RUN_MODE.
	RunMode string `mapstructure:"run_mode"`
	// OllamaVersion là phiên bản Ollama được proxy công bố qua /api/version.
	OllamaVersion string `mapstructure:"ollama_version"`
	// LlamaServerHost và LlamaServerPort được sử dụng bởi chế độ wrapper khi tự động khởi chạy llama-server.
	LlamaServerHost string `mapstructure:"llama_server_host"`
	LlamaServerPort string `mapstructure:"llama_server_port"`
}

// Server is an Ollama-compatible HTTP proxy server.
type Server struct {
	echo *echo.Echo
	cfg  Config
}

// New tạo mới Ollama proxy Server từ cfg.
// cfg.Host và cfg.Port sẽ dùng giá trị mặc định OllamaProxyHost/OllamaProxyPort nếu để trống.
func New(cfg Config) (*Server, error) {
	if cfg.Host == "" {
		cfg.Host = OllamaProxyHost
	}
	if cfg.Port == "" {
		cfg.Port = OllamaProxyPort
	}
	if cfg.OllamaVersion == "" {
		cfg.OllamaVersion = OllamaVersion
	}

	h, err := newHandlers(cfg.ModelName, cfg.ModelSize, cfg.UpstreamBase, cfg.OllamaVersion)
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

// Start chạy proxy server trên cfg.Host:cfg.Port.
// Blocks cho đến khi ctx bị huỷ, sau đó thực hiện graceful shutdown (timeout 10s).
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
