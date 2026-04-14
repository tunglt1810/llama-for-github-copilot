// Package proxy implements an Ollama-compatible HTTP API that proxies requests
// to a running llama-server instance (OpenAI-compatible API).
//
// Supported endpoints (Ollama API v0.20.7):
//
//	GET  /api/version              → {"version":"0.20.7"}
//	GET  /api/tags                 → danh sách models (Ollama format)
//	GET  /api/ps                   → models đang chạy
//	POST /api/chat                 → translate Ollama→OpenAI, proxy, translate back
//	POST /api/generate             → translate Ollama→OpenAI /v1/completions, proxy, translate back
//	POST /api/embed                → translate Ollama→OpenAI /v1/embeddings, proxy, translate back
//	POST /api/show                 → model metadata
//	GET  /v1/models                → served locally
//	ANY  /v1/*                     → pass-through to llama-server
package proxy

import (
	"encoding/json"
	"time"
)

// ─── Ollama version ──────────────────────────────────────────────────────────

// OllamaVersion phải khớp với phiên bản Ollama mà client (GitHub Copilot, Continue.dev, v.v.)
// yêu cầu tối thiểu để enable integration.
const OllamaVersion = "0.20.7"

type VersionResponse struct {
	Version string `json:"version"`
}

// ─── Ollama model list (GET /api/tags) ───────────────────────────────────────

type ModelDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

type ModelInfo struct {
	Name       string       `json:"name"`
	Model      string       `json:"model"`
	ModifiedAt time.Time    `json:"modified_at"`
	Size       int64        `json:"size"`
	Digest     string       `json:"digest"`
	Details    ModelDetails `json:"details"`
}

type TagsResponse struct {
	Models []ModelInfo `json:"models"`
}

// ─── Ollama running models (GET /api/ps) ─────────────────────────────────────

type RunningModel struct {
	Name      string       `json:"name"`
	Model     string       `json:"model"`
	Size      int64        `json:"size"`
	Digest    string       `json:"digest"`
	Details   ModelDetails `json:"details"`
	ExpiresAt time.Time    `json:"expires_at"`
	SizeVRAM  int64        `json:"size_vram"`
}

type PSResponse struct {
	Models []RunningModel `json:"models"`
}

// ─── Ollama show (POST /api/show) ─────────────────────────────────────────────

type ShowRequest struct {
	Model   string `json:"model"`
	Verbose bool   `json:"verbose,omitempty"`
}

type ShowResponse struct {
	License      string              `json:"license,omitempty"`
	Modelfile    string              `json:"modelfile"`
	Parameters   string              `json:"parameters"`
	Template     string              `json:"template"`
	Details      ModelDetails        `json:"details"`
	ModelInfo    map[string]any      `json:"model_info,omitempty"`
	ModifiedAt   time.Time           `json:"modified_at"`
	// Capabilities liệt kê khả năng của model; GitHub Copilot filter dựa trên field này.
	// Phải có ít nhất "completion" để Copilot hiển thị model trong danh sách.
	Capabilities []string            `json:"capabilities"`
}

// ─── Tool types (Ollama) ──────────────────────────────────────────────────────

// ToolFunction mô tả function trong một Tool.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Tool biểu diễn một tool (function) gửi từ client đến /api/chat.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolCallFunction là kết quả function call từ model.
type ToolCallFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ToolCall biểu diễn một lần model gọi tool.
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

// ─── Ollama chat (POST /api/chat) ─────────────────────────────────────────────

// OllamaMessage là một message trong chat history.
type OllamaMessage struct {
	Role string `json:"role"`
	// Content là nội dung text của message.
	Content string `json:"content"`
	// Images là danh sách ảnh base64-encoded (dành cho multimodal models như llava).
	Images []string `json:"images,omitempty"`
	// ToolCalls là danh sách tool calls mà model muốn thực hiện.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolName là tên tool đã được thực thi (dùng cho role: "tool").
	ToolName string `json:"tool_name,omitempty"`
	// Thinking là nội dung "thinking" của model (cho các thinking models như deepseek-r1).
	Thinking string `json:"thinking,omitempty"`
}

type ChatOptions struct {
	Temperature   *float64 `json:"temperature,omitempty"`
	TopP          *float64 `json:"top_p,omitempty"`
	TopK          *int     `json:"top_k,omitempty"`
	NumPredict    *int     `json:"num_predict,omitempty"`
	RepeatPenalty *float64 `json:"repeat_penalty,omitempty"`
	Seed          *int     `json:"seed,omitempty"`
}

type ChatRequest struct {
	Model    string          `json:"model"`
	Messages []OllamaMessage `json:"messages"`
	Stream   *bool           `json:"stream,omitempty"` // nil nghĩa là mặc định (true)
	Options  *ChatOptions    `json:"options,omitempty"`
	// Tools là danh sách tools mà model có thể sử dụng (function calling).
	Tools []Tool `json:"tools,omitempty"`
	// Format là định dạng output: "json" hoặc JSON schema object cho structured outputs.
	Format json.RawMessage `json:"format,omitempty"`
	// KeepAlive kiểm soát thời gian model giữ trong memory (mặc định "5m").
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
	// Think cho phép model "suy nghĩ" trước khi trả lời (dành cho thinking models).
	Think *bool `json:"think,omitempty"`
}

// ChatResponse là một dòng NDJSON trong streaming hoặc body đầy đủ khi không streaming.
type ChatResponse struct {
	Model     string        `json:"model"`
	CreatedAt time.Time     `json:"created_at"`
	Message   OllamaMessage `json:"message"`
	Done      bool          `json:"done"`

	// Các field dưới đây chỉ có trong chunk cuối cùng (done: true)
	DoneReason         string `json:"done_reason,omitempty"`
	TotalDuration      int64  `json:"total_duration,omitempty"`
	LoadDuration       int64  `json:"load_duration,omitempty"`
	PromptEvalCount    int    `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64  `json:"prompt_eval_duration,omitempty"`
	EvalCount          int    `json:"eval_count,omitempty"`
	EvalDuration       int64  `json:"eval_duration,omitempty"`
}

// ─── Ollama generate (POST /api/generate) ────────────────────────────────────

// GenerateRequest là request body cho POST /api/generate (text completion đơn lượt).
type GenerateRequest struct {
	Model    string          `json:"model"`
	Prompt   string          `json:"prompt"`
	Suffix   string          `json:"suffix,omitempty"`
	Images   []string        `json:"images,omitempty"`
	System   string          `json:"system,omitempty"`
	Template string          `json:"template,omitempty"`
	Stream   *bool           `json:"stream,omitempty"`
	Raw      bool            `json:"raw,omitempty"`
	Format   json.RawMessage `json:"format,omitempty"`
	// KeepAlive kiểm soát thời gian model giữ trong memory.
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
	// Think cho phép model "suy nghĩ" trước khi trả lời.
	Think   *bool        `json:"think,omitempty"`
	Options *ChatOptions `json:"options,omitempty"`
}

// GenerateResponse là một dòng NDJSON khi streaming hoặc body đầy đủ khi không streaming.
type GenerateResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Response  string    `json:"response"`
	Done      bool      `json:"done"`

	// Thinking là nội dung "thinking" của model (cho thinking models).
	Thinking string `json:"thinking,omitempty"`

	// Các field dưới đây chỉ có trong response cuối (done: true)
	DoneReason         string `json:"done_reason,omitempty"`
	Context            []int  `json:"context,omitempty"`
	TotalDuration      int64  `json:"total_duration,omitempty"`
	LoadDuration       int64  `json:"load_duration,omitempty"`
	PromptEvalCount    int    `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64  `json:"prompt_eval_duration,omitempty"`
	EvalCount          int    `json:"eval_count,omitempty"`
	EvalDuration       int64  `json:"eval_duration,omitempty"`
}

// ─── Ollama embed (POST /api/embed) ──────────────────────────────────────────

// EmbedRequest là request body cho POST /api/embed.
// Input có thể là một string hoặc []string — dùng json.RawMessage để handle cả hai.
type EmbedRequest struct {
	Model    string          `json:"model"`
	Input    json.RawMessage `json:"input"`
	Truncate *bool           `json:"truncate,omitempty"`
	// KeepAlive kiểm soát thời gian model giữ trong memory.
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
	Options   *ChatOptions    `json:"options,omitempty"`
}

// EmbedResponse là response cho POST /api/embed.
type EmbedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float64 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration,omitempty"`
	LoadDuration    int64       `json:"load_duration,omitempty"`
	PromptEvalCount int         `json:"prompt_eval_count,omitempty"`
}

// ─── OpenAI internal types (dùng bởi translate.go) ───────────────────────────

// openAIResponseFormat dùng cho structured outputs và JSON mode.
type openAIResponseFormat struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"json_schema,omitempty"`
}

// openAIToolFunction mô tả function trong openAITool.
type openAIToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// openAITool là OpenAI tool format.
type openAITool struct {
	Type     string             `json:"type"`
	Function openAIToolFunction `json:"function"`
}

// openAIToolCallFunction là function call result từ model.
type openAIToolCallFunction struct {
	Name string `json:"name"`
	// Arguments là JSON-encoded string (khác với Ollama dùng json.RawMessage object).
	Arguments string `json:"arguments"`
}

// openAIToolCall biểu diễn một tool call từ model.
type openAIToolCall struct {
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function openAIToolCallFunction `json:"function"`
}

// openAIMessage là một message trong OpenAI chat format.
// Content dùng json.RawMessage để hỗ trợ cả string thường và content array (multimodal).
type openAIMessage struct {
	Role       string           `json:"role"`
	Content    json.RawMessage  `json:"content,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	// Name là tên tool cho role "tool" messages.
	Name string `json:"name,omitempty"`
}

type openAIChatRequest struct {
	Model          string                `json:"model"`
	Messages       []openAIMessage       `json:"messages"`
	Stream         bool                  `json:"stream"`
	Temperature    *float64              `json:"temperature,omitempty"`
	TopP           *float64              `json:"top_p,omitempty"`
	MaxTokens      *int                  `json:"max_tokens,omitempty"`
	Seed           *int                  `json:"seed,omitempty"`
	Tools          []openAITool          `json:"tools,omitempty"`
	ResponseFormat *openAIResponseFormat `json:"response_format,omitempty"`
}

// openAIStreamChunk là SSE data payload từ OpenAI streaming API.
type openAIStreamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
			// ToolCalls trong streaming được gửi từng phần — cần accumulate theo index.
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id,omitempty"`
				Type     string `json:"type,omitempty"`
				Function struct {
					Name      string `json:"name,omitempty"`
					Arguments string `json:"arguments,omitempty"`
				} `json:"function,omitempty"`
			} `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}

// openAIChatResponse là non-streaming OpenAI response.
type openAIChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int           `json:"index"`
		Message      openAIMessage `json:"message"`
		FinishReason string        `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// ─── OpenAI Completions internal types (dùng cho /api/generate) ──────────────

// openAICompletionRequest là request body cho OpenAI /v1/completions.
type openAICompletionRequest struct {
	Model          string                `json:"model"`
	Prompt         string                `json:"prompt"`
	Suffix         string                `json:"suffix,omitempty"`
	Stream         bool                  `json:"stream"`
	Temperature    *float64              `json:"temperature,omitempty"`
	TopP           *float64              `json:"top_p,omitempty"`
	MaxTokens      *int                  `json:"max_tokens,omitempty"`
	Seed           *int                  `json:"seed,omitempty"`
	ResponseFormat *openAIResponseFormat `json:"response_format,omitempty"`
}

// openAICompletionResponse là non-streaming /v1/completions response.
type openAICompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		Text         string `json:"text"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// openAICompletionStreamChunk là streaming chunk từ /v1/completions.
type openAICompletionStreamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int     `json:"index"`
		Text         string  `json:"text"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

// ─── OpenAI Embeddings internal types (dùng cho /api/embed) ──────────────────

// openAIEmbedRequest là request body cho OpenAI /v1/embeddings.
type openAIEmbedRequest struct {
	Model string          `json:"model"`
	Input json.RawMessage `json:"input"`
}

// openAIEmbedResponse là response từ OpenAI /v1/embeddings.
type openAIEmbedResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}
