// Package proxy implements an Ollama-compatible HTTP API that proxies requests
// to a running llama-server instance (OpenAI-compatible API).
//
// Supported endpoints (Ollama API v0.20.7):

// GET  /api/version              → {"version":"0.20.7"}
// GET  /api/tags                 → list of models (Ollama format)
// GET  /api/ps                   → running models
// POST /api/chat                 → translate Ollama→OpenAI, proxy, translate back
// POST /api/generate             → translate Ollama→OpenAI /v1/completions, proxy, translate back
// POST /api/embed                → translate Ollama→OpenAI /v1/embeddings, proxy, translate back
// POST /api/show                 → model metadata
// GET  /v1/models                → served locally
// ANY  /v1/*                     → pass-through to llama-server
package proxy

import (
	"encoding/json"
	"time"
)

// ─── Ollama version ──────────────────────────────────────────────────────────

// OllamaVersion must match the minimum Ollama version required by clients
// (GitHub Copilot, Continue.dev, etc.) to enable integration.
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
	// Capabilities lists the model's capabilities; GitHub Copilot filters based on this field.
	// The field must include at least "completion" for Copilot to display the model in lists.
	Capabilities []string            `json:"capabilities"`
}

// ─── Tool types (Ollama) ──────────────────────────────────────────────────────

// ToolFunction describes a function inside a Tool.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Tool represents a tool (function) sent by the client to /api/chat.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolCallFunction is the result of a function call from the model.
type ToolCallFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ToolCall represents a single tool call from the model.
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

// ─── Ollama chat (POST /api/chat) ─────────────────────────────────────────────

// OllamaMessage is a message in the chat history.
type OllamaMessage struct {
	Role string `json:"role"`
	// Content is the text content of the message.
	Content string `json:"content"`
	// Images is a list of base64-encoded images (for multimodal models like llava).
	Images []string `json:"images,omitempty"`
	// ToolCalls is the list of tool calls the model wants to perform.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolName is the name of the executed tool (used for role: "tool").
	ToolName string `json:"tool_name,omitempty"`
	// Thinking is the model's "thinking" content (for thinking models like deepseek-r1).
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
	Stream   *bool           `json:"stream,omitempty"` // nil means default (true)
	Options  *ChatOptions    `json:"options,omitempty"`
	// Tools is the list of tools that the model can use (function calling).
	Tools []Tool `json:"tools,omitempty"`
	// Format is the output format: "json" or a JSON schema object for structured outputs.
	Format json.RawMessage `json:"format,omitempty"`
	// KeepAlive controls how long the model is kept in memory (default "5m").
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
	// Think allows the model to "think" before responding (for thinking models).
	Think *bool `json:"think,omitempty"`
}

// ChatResponse is an NDJSON line when streaming, or a full body when not streaming.
type ChatResponse struct {
	Model     string        `json:"model"`
	CreatedAt time.Time     `json:"created_at"`
	Message   OllamaMessage `json:"message"`
	Done      bool          `json:"done"`

	// The fields below are only present in the final chunk (done: true)
	DoneReason         string `json:"done_reason,omitempty"`
	TotalDuration      int64  `json:"total_duration,omitempty"`
	LoadDuration       int64  `json:"load_duration,omitempty"`
	PromptEvalCount    int    `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64  `json:"prompt_eval_duration,omitempty"`
	EvalCount          int    `json:"eval_count,omitempty"`
	EvalDuration       int64  `json:"eval_duration,omitempty"`
}

// ─── Ollama generate (POST /api/generate) ────────────────────────────────────

// GenerateRequest is the request body for POST /api/generate (single-turn text completion).
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
	// KeepAlive controls how long the model is kept in memory.
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
	// Think allows the model to "think" before responding.
	Think   *bool        `json:"think,omitempty"`
	Options *ChatOptions `json:"options,omitempty"`
}

// GenerateResponse is an NDJSON line when streaming or a full body when not streaming.
type GenerateResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Response  string    `json:"response"`
	Done      bool      `json:"done"`

	// Thinking is the model's "thinking" content (for thinking models).
	Thinking string `json:"thinking,omitempty"`

	// The fields below are only present in the final response (done: true)
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

// EmbedRequest is the request body for POST /api/embed.
// Input can be a string or []string — json.RawMessage is used to handle both.
type EmbedRequest struct {
	Model    string          `json:"model"`
	Input    json.RawMessage `json:"input"`
	Truncate *bool           `json:"truncate,omitempty"`
	// KeepAlive controls how long the model is kept in memory.
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
	Options   *ChatOptions    `json:"options,omitempty"`
}

// EmbedResponse is the response for POST /api/embed.
type EmbedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float64 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration,omitempty"`
	LoadDuration    int64       `json:"load_duration,omitempty"`
	PromptEvalCount int         `json:"prompt_eval_count,omitempty"`
}

// ─── OpenAI internal types (used by translate.go) ───────────────────────────

// openAIResponseFormat is used for structured outputs and JSON mode.
type openAIResponseFormat struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"json_schema,omitempty"`
}

// openAIToolFunction describes a function in an openAITool.
type openAIToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// openAITool is the OpenAI tool format.
type openAITool struct {
	Type     string             `json:"type"`
	Function openAIToolFunction `json:"function"`
}

// openAIToolCallFunction là function call result từ model.
type openAIToolCallFunction struct {
	Name string `json:"name"`
	// Arguments is a JSON-encoded string (unlike Ollama which uses json.RawMessage).
	Arguments string `json:"arguments"`
}

// openAIToolCall represents a tool call from the model.
type openAIToolCall struct {
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function openAIToolCallFunction `json:"function"`
}

// openAIMessage is a message in the OpenAI chat format.
// Content uses json.RawMessage to support both plain strings and content arrays (multimodal).
type openAIMessage struct {
	Role       string           `json:"role"`
	Content    json.RawMessage  `json:"content,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	// Name is the tool name for role "tool" messages.
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

// openAIStreamChunk is the SSE data payload from the OpenAI streaming API.
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
			// ToolCalls in streaming are sent in parts — they need to be accumulated by index.
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

// openAIChatResponse is a non-streaming OpenAI response.
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

// ─── OpenAI Completions internal types (used for /api/generate) ──────────────

// openAICompletionRequest is the request body for OpenAI /v1/completions.
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

// openAICompletionResponse is a non-streaming /v1/completions response.
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

// openAICompletionStreamChunk is a streaming chunk from /v1/completions.
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

// ─── OpenAI Embeddings internal types (used for /api/embed) ──────────────────

// openAIEmbedRequest is the request body for OpenAI /v1/embeddings.
type openAIEmbedRequest struct {
	Model string          `json:"model"`
	Input json.RawMessage `json:"input"`
}

// openAIEmbedResponse is the response from OpenAI /v1/embeddings.
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
