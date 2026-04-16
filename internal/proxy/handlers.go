package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"regexp"
	"strconv"
	"time"

	"github.com/labstack/echo/v5"
	"github.com/rs/zerolog/log"
)

// handlers groups all Ollama-compatible endpoint logic.
// modelName is the active model alias; upstreamBase is the llama-server base URL.
type handlers struct {
	modelName     string
	modelSize     int64
	upstreamBase  string
	reverseProxy  *httputil.ReverseProxy
	ollamaVersion string
	contextSize   int
}

func newHandlers(modelName string, modelSize int64, upstreamBase string, ollamaVersion string, contextSize int) (*handlers, error) {
	target, err := url.Parse(upstreamBase)
	if err != nil {
		return nil, fmt.Errorf("parsing upstream URL: %w", err)
	}
	rp := httputil.NewSingleHostReverseProxy(target)
	// SECURITY: suppress internal error details from client responses
	rp.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Error().Err(err).Str("path", r.URL.Path).Msg("reverse proxy error")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "upstream unavailable"})
	}
	return &handlers{
		modelName:     modelName,
		modelSize:     modelSize,
		upstreamBase:  upstreamBase,
		reverseProxy:  rp,
		ollamaVersion: ollamaVersion,
		contextSize:   contextSize,
	}, nil
}

// ─── GET /api/version ─────────────────────────────────────────────────────────

func (h *handlers) version(c *echo.Context) error {
	return c.JSON(http.StatusOK, VersionResponse{Version: h.ollamaVersion})
}

// ─── GET /api/tags ────────────────────────────────────────────────────────────

func (h *handlers) tags(c *echo.Context) error {
	return c.JSON(http.StatusOK, TagsResponse{
		Models: []ModelInfo{h.toModelInfo()},
	})
}

// ─── GET /api/ps ──────────────────────────────────────────────────────────────

func (h *handlers) ps(c *echo.Context) error {
	rm := RunningModel{
		Name:      h.modelName,
		Model:     h.modelName,
		Size:      h.modelSize,
		Details:   h.modelDetails(),
		ExpiresAt: time.Now().Add(5 * time.Minute),
		SizeVRAM:  h.modelSize,
	}
	return c.JSON(http.StatusOK, PSResponse{Models: []RunningModel{rm}})
}

// ─── POST /api/show ───────────────────────────────────────────────────────────

const showTemplate = `{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>`

func (h *handlers) show(c *echo.Context) error {
	var req ShowRequest
	if err := c.Bind(&req); err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid request body")
	}

	resp := ShowResponse{
		License:      "",
		Modelfile:    fmt.Sprintf("FROM \"%s\"\nPARAMETER num_ctx %d\n", h.modelName, h.contextLength()),
		Parameters:   "stop                           \"<|start_header_id|>\"\nstop                           \"<|end_header_id|>\"\nstop                           \"<|eot_id|>\"\n",
		Template:     showTemplate,
		Details:      h.modelDetails(),
		ModifiedAt:   time.Now(),
		Capabilities: h.showCapabilities(),
		ModelInfo:    h.modelInfo(),
	}

	if req.Verbose {
		resp.ModelInfo = h.modelInfo()
	}

	return c.JSON(http.StatusOK, resp)
}

// ─── POST /api/chat ───────────────────────────────────────────────────────────

func (h *handlers) chat(c *echo.Context) error {
	var req ChatRequest
	if err := c.Bind(&req); err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid request body")
	}

	// Use the running model if client sends a generic name
	if req.Model == "" {
		req.Model = h.modelName
	}

	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	// Build OpenAI request
	openAIReq := ollamaToOpenAI(&req)
	body, contentLen, err := marshalBody(openAIReq)
	if err != nil {
		return echo.NewHTTPError(http.StatusInternalServerError, "failed to encode request")
	}

	// Forward to llama-server
	upstream := h.upstreamBase + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(c.Request().Context(), http.MethodPost, upstream, body)
	if err != nil {
		return echo.NewHTTPError(http.StatusInternalServerError, "failed to build upstream request")
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.ContentLength = contentLen

	client := &http.Client{Timeout: 0} // streaming needs no read timeout
	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error().Err(err).Msg("upstream request failed")
		return echo.NewHTTPError(http.StatusBadGateway, "upstream unavailable")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Relay upstream error body as-is (generic status to client)
		log.Error().Int("status", resp.StatusCode).Msg("upstream returned error")
		return c.NoContent(http.StatusBadGateway)
	}

	if stream {
		c.Response().Header().Set("Content-Type", "application/x-ndjson")
		c.Response().WriteHeader(http.StatusOK)
		if err := streamOpenAIToOllama(c.Response(), resp.Body, h.modelName); err != nil {
			log.Error().Err(err).Msg("stream translation error")
		}
		return nil
	}

	// Non-streaming: decode OpenAI response and translate
	var oaiResp openAIChatResponse
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return echo.NewHTTPError(http.StatusBadGateway, "failed to read upstream response")
	}
	if err := json.Unmarshal(data, &oaiResp); err != nil {
		return echo.NewHTTPError(http.StatusBadGateway, "failed to parse upstream response")
	}
	ollamaResp := openAIToOllamaResponse(h.modelName, &oaiResp)
	return c.JSON(http.StatusOK, ollamaResp)
}

// ─── Passthrough for /v1/* ────────────────────────────────────────────────────

// passthrough forwards the request verbatim to llama-server's /v1/ API.
// This exposes the OpenAI-compatible endpoint alongside the Ollama one.
func (h *handlers) passthrough(c *echo.Context) error {
	h.reverseProxy.ServeHTTP(c.Response(), c.Request())
	return nil
}

// ─── GET /v1/models ───────────────────────────────────────────────────────────

// v1Models natively handles model discovery for clients utilizing the OpenAI standard
// instead of sending it upstream, as llama-server might return non-friendly IDs.
func (h *handlers) v1Models(c *echo.Context) error {
	type openaiModel struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	type openaiModelsResponse struct {
		Object string        `json:"object"`
		Data   []openaiModel `json:"data"`
	}
	return c.JSON(http.StatusOK, openaiModelsResponse{
		Object: "list",
		Data: []openaiModel{
			{
				ID:      h.modelName,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "system",
			},
		},
	})
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

func (h *handlers) toModelInfo() ModelInfo {
	return ModelInfo{
		Name:       h.modelName,
		Model:      h.modelName,
		ModifiedAt: time.Now(),
		Size:       h.modelSize,
		Digest:     "sha256:" + h.modelName,
		Details:    h.modelDetails(),
	}
}

func (h *handlers) modelDetails() ModelDetails {
	details := ModelDetails{
		Format:   "gguf",
		Family:   "llama",
		Families: []string{"llama"},
	}

	if h.modelSize > 0 {
		details.ParameterSize = fmt.Sprintf("%.1fB", float64(h.modelSize)/1e9)
	} else if size := h.guessParameterCount(); size > 0 {
		details.ParameterSize = fmt.Sprintf("%.1fB", float64(size)/1e9)
	}

	return details
}

func (h *handlers) contextLength() int {
	if h.contextSize > 0 {
		return h.contextSize
	}
	return 262144
}

func (h *handlers) showCapabilities() []string {
	caps := []string{"completion", "chat", "tools", "vision"}
	return caps
}

func (h *handlers) modelInfo() map[string]any {
	info := map[string]any{
		"general.architecture":         "llama",
		"general.file_type":            2,
		"general.quantization_version": 2,
		"general.context_length":       h.contextLength(),
		"llama.context_length":         h.contextLength(),
		"tokenizer.ggml.model":         "gpt2",
		"tokenizer.ggml.pre":           "llama-bpe",
	}

	if h.modelSize > 0 {
		info["general.parameter_count"] = h.modelSize
	} else if count := h.guessParameterCount(); count > 0 {
		info["general.parameter_count"] = count
	}

	return info
}

func (h *handlers) guessParameterCount() int64 {
	re := regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*b`)
	match := re.FindStringSubmatch(h.modelName)
	if len(match) < 2 {
		return 0
	}

	num, err := strconv.ParseFloat(match[1], 64)
	if err != nil {
		return 0
	}
	return int64(num * 1e9)
}

// ─── POST /api/generate ───────────────────────────────────────────────────────

func (h *handlers) generate(c *echo.Context) error {
	var req GenerateRequest
	if err := c.Bind(&req); err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid request body")
	}
	if req.Model == "" {
		req.Model = h.modelName
	}

	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	oaiReq := ollamaGenerateToOpenAI(&req)
	body, contentLen, err := marshalBody(oaiReq)
	if err != nil {
		return echo.NewHTTPError(http.StatusInternalServerError, "failed to encode request")
	}

	upstream := h.upstreamBase + "/v1/completions"
	httpReq, err := http.NewRequestWithContext(c.Request().Context(), http.MethodPost, upstream, body)
	if err != nil {
		return echo.NewHTTPError(http.StatusInternalServerError, "failed to build upstream request")
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.ContentLength = contentLen

	client := &http.Client{Timeout: 0}
	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error().Err(err).Msg("upstream generate request failed")
		return echo.NewHTTPError(http.StatusBadGateway, "upstream unavailable")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Error().Int("status", resp.StatusCode).Msg("upstream generate returned error")
		return c.NoContent(http.StatusBadGateway)
	}

	if stream {
		c.Response().Header().Set("Content-Type", "application/x-ndjson")
		c.Response().WriteHeader(http.StatusOK)
		if err := streamOpenAIGenerateToOllama(c.Response(), resp.Body, h.modelName); err != nil {
			log.Error().Err(err).Msg("generate stream translation error")
		}
		return nil
	}

	var oaiResp openAICompletionResponse
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return echo.NewHTTPError(http.StatusBadGateway, "failed to read upstream response")
	}
	if err := json.Unmarshal(data, &oaiResp); err != nil {
		return echo.NewHTTPError(http.StatusBadGateway, "failed to parse upstream response")
	}
	return c.JSON(http.StatusOK, openAIToOllamaGenerate(h.modelName, &oaiResp))
}

// ─── POST /api/embed ──────────────────────────────────────────────────────────

func (h *handlers) embed(c *echo.Context) error {
	var req EmbedRequest
	if err := c.Bind(&req); err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid request body")
	}
	if req.Model == "" {
		req.Model = h.modelName
	}

	oaiReq := ollamaEmbedToOpenAI(&req)
	body, contentLen, err := marshalBody(oaiReq)
	if err != nil {
		return echo.NewHTTPError(http.StatusInternalServerError, "failed to encode request")
	}

	upstream := h.upstreamBase + "/v1/embeddings"
	httpReq, err := http.NewRequestWithContext(c.Request().Context(), http.MethodPost, upstream, body)
	if err != nil {
		return echo.NewHTTPError(http.StatusInternalServerError, "failed to build upstream request")
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.ContentLength = contentLen

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		log.Error().Err(err).Msg("upstream embed request failed")
		return echo.NewHTTPError(http.StatusBadGateway, "upstream unavailable")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Error().Int("status", resp.StatusCode).Msg("upstream embed returned error")
		return c.NoContent(http.StatusBadGateway)
	}

	var oaiResp openAIEmbedResponse
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return echo.NewHTTPError(http.StatusBadGateway, "failed to read upstream response")
	}
	if err := json.Unmarshal(data, &oaiResp); err != nil {
		return echo.NewHTTPError(http.StatusBadGateway, "failed to parse upstream response")
	}
	return c.JSON(http.StatusOK, openAIToOllamaEmbed(h.modelName, &oaiResp))
}
