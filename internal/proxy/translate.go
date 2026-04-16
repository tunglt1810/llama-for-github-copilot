package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// ─── Common helpers ───────────────────────────────────────────────────────

// marshalBody serializes v into JSON and wraps it in an io.ReadCloser suitable for http.Request.Body.
func marshalBody(v any) (io.ReadCloser, int64, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return nil, 0, err
	}
	return io.NopCloser(bytes.NewReader(data)), int64(len(data)), nil
}

// toOpenAIContent converts text and images into a json.RawMessage compatible with the OpenAI format.
// If there are no images: returns a JSON string.
// If there are images: returns a JSON array of content parts (text + image_url).
func toOpenAIContent(text string, images []string) json.RawMessage {
	if len(images) == 0 {
		b, _ := json.Marshal(text)
		return json.RawMessage(b)
	}
	// Use anonymous structs to avoid adding extra imports
	type textPart struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	type imageURL struct {
		URL string `json:"url"`
	}
	type imagePart struct {
		Type     string   `json:"type"`
		ImageURL imageURL `json:"image_url"`
	}
	parts := make([]any, 0, 1+len(images))
	if text != "" {
		parts = append(parts, textPart{Type: "text", Text: text})
	}
	for _, img := range images {
		url := img
		// If the data URL prefix is missing, add it
		if !strings.HasPrefix(img, "data:") {
			url = "data:image/png;base64," + img
		}
		parts = append(parts, imagePart{Type: "image_url", ImageURL: imageURL{URL: url}})
	}
	b, _ := json.Marshal(parts)
	return json.RawMessage(b)
}

// extractTextContent extracts text content from an OpenAI content field (json.RawMessage).
// Supports both a string and an array of content parts.
func extractTextContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	// Try parsing as a string first
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	// Then try an array of parts — collect all text parts
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var sb strings.Builder
		for _, p := range parts {
			if p.Type == "text" {
				sb.WriteString(p.Text)
			}
		}
		return sb.String()
	}
	return ""
}

// ollamaFormatToOpenAI converts the Ollama 'format' field to OpenAI 'response_format'.
// "json" → {"type":"json_object"}
// JSON schema object → {"type":"json_schema","json_schema":<schema>}
func ollamaFormatToOpenAI(format json.RawMessage) *openAIResponseFormat {
	if len(format) == 0 {
		return nil
	}
	// Check whether it is the string "json"
	var s string
	if err := json.Unmarshal(format, &s); err == nil {
		if s == "json" {
			return &openAIResponseFormat{Type: "json_object"}
		}
		return nil
	}
	// It's a JSON object (JSON schema) → use json_schema type
	return &openAIResponseFormat{Type: "json_schema", Schema: format}
}

// ─── Chat: Ollama → OpenAI ────────────────────────────────────────────────────

// ollamaToOpenAI converts an Ollama POST /api/chat request into an OpenAI
// POST /v1/chat/completions request body.
func ollamaToOpenAI(req *ChatRequest) *openAIChatRequest {
	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	out := &openAIChatRequest{
		Model:  req.Model,
		Stream: stream,
	}

	// Convert messages — supports images, tool_calls, and tool role
	out.Messages = make([]openAIMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		msg := openAIMessage{
			Role:    m.Role,
			Content: toOpenAIContent(m.Content, m.Images),
		}
		// Convert Ollama tool_calls (arguments are json.RawMessage objects) into
		// OpenAI tool_calls (arguments are JSON-encoded strings)
		if len(m.ToolCalls) > 0 {
			msg.ToolCalls = make([]openAIToolCall, len(m.ToolCalls))
			for i, tc := range m.ToolCalls {
				argsStr := string(tc.Function.Arguments)
				if argsStr == "" || argsStr == "null" {
					argsStr = "{}"
				}
				msg.ToolCalls[i] = openAIToolCall{
					Type: "function",
					Function: openAIToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: argsStr,
					},
				}
			}
			// When there are tool_calls, content is typically null/empty
			msg.Content = toOpenAIContent("", nil)
		}
		// Handle role "tool" — map tool_name to the name field
		if m.Role == "tool" && m.ToolName != "" {
			msg.Name = m.ToolName
			// Use tool_name as a fake tool_call_id because Ollama doesn't provide this field
			msg.ToolCallID = "call_" + m.ToolName
		}
		out.Messages = append(out.Messages, msg)
	}

	// Flatten options
	if o := req.Options; o != nil {
		out.Temperature = o.Temperature
		out.TopP = o.TopP
		out.MaxTokens = o.NumPredict
		out.Seed = o.Seed
	}

	// Convert tools
	if len(req.Tools) > 0 {
		out.Tools = make([]openAITool, len(req.Tools))
		for i, t := range req.Tools {
			out.Tools[i] = openAITool{
				Type: t.Type,
				Function: openAIToolFunction{
					Name:        t.Function.Name,
					Description: t.Function.Description,
					Parameters:  t.Function.Parameters,
				},
			}
		}
	}

	// Convert format → response_format
	out.ResponseFormat = ollamaFormatToOpenAI(req.Format)

	return out
}

// ─── Chat: OpenAI → Ollama (non-streaming) ────────────────────────────────────

// openAIToOllamaResponse converts a non-streaming OpenAI response into an Ollama ChatResponse.
func openAIToOllamaResponse(model string, oai *openAIChatResponse) *ChatResponse {
	content := ""
	var toolCalls []ToolCall
	finishReason := "stop"

	if len(oai.Choices) > 0 {
		choice := oai.Choices[0]
		content = extractTextContent(choice.Message.Content)
		if choice.FinishReason != "" {
			finishReason = choice.FinishReason
		}
		// Convert OpenAI tool_calls (arguments as strings) to Ollama (arguments as json.RawMessage)
		if len(choice.Message.ToolCalls) > 0 {
			toolCalls = make([]ToolCall, 0, len(choice.Message.ToolCalls))
			for _, tc := range choice.Message.ToolCalls {
				var args json.RawMessage
				if json.Valid([]byte(tc.Function.Arguments)) {
					args = json.RawMessage(tc.Function.Arguments)
				} else {
					args = json.RawMessage("{}")
				}
				toolCalls = append(toolCalls, ToolCall{
					Function: ToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: args,
					},
				})
			}
		}
	}

	return &ChatResponse{
		Model:     model,
		CreatedAt: time.Now(),
		Message: OllamaMessage{
			Role:      "assistant",
			Content:   content,
			ToolCalls: toolCalls,
		},
		Done:            true,
		DoneReason:      finishReason,
		PromptEvalCount: oai.Usage.PromptTokens,
		EvalCount:       oai.Usage.CompletionTokens,
	}
}

// ─── Chat: OpenAI → Ollama (streaming) ────────────────────────────────────────

// pendingToolCall accumulates streaming tool_call chunks by index.
type pendingToolCall struct {
	id        string
	name      string
	arguments strings.Builder
}

// streamOpenAIToOllama reads an OpenAI SSE stream from src, converts each chunk to
// Ollama NDJSON format, and writes to dst.
//
// Supports: text content, tool_calls (accumulate partial chunks).
//
// OpenAI SSE format:
//
//	data: {"choices":[{"delta":{"content":"TOKEN"},...}],...}
//	data: [DONE]
//
// Ollama NDJSON format:
//
//	{"model":"...","created_at":"...","message":{"role":"assistant","content":"TOKEN"},"done":false}
//	{"model":"...","created_at":"...","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}
func streamOpenAIToOllama(dst io.Writer, src io.Reader, model string) error {
	scanner := bufio.NewScanner(src)
	// Increase buffer size to handle large chunks
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	enc := json.NewEncoder(dst)

	// Map to accumulate tool_calls by index
	pendingCalls := map[int]*pendingToolCall{}

	flushFn := func() {
		if f, ok := dst.(interface{ Flush() }); ok {
			f.Flush()
		}
	}

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")

		if payload == "[DONE]" {
			// If there are pending tool_calls not yet emitted (due to missing finish_reason chunk)
			if len(pendingCalls) > 0 {
				_ = emitToolCallChunk(enc, model, pendingCalls, "tool_calls")
				pendingCalls = map[int]*pendingToolCall{}
				flushFn()
			}
			// Emit final done chunk
			final := ChatResponse{
				Model:      model,
				CreatedAt:  time.Now(),
				Message:    OllamaMessage{Role: "assistant", Content: ""},
				Done:       true,
				DoneReason: "stop",
			}
			_ = enc.Encode(final)
			flushFn()
			return nil
		}

		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		delta := choice.Delta

		// Accumulate tool_calls if present
		for _, tc := range delta.ToolCalls {
			if pendingCalls[tc.Index] == nil {
				pendingCalls[tc.Index] = &pendingToolCall{}
			}
			p := pendingCalls[tc.Index]
			if tc.ID != "" {
				p.id = tc.ID
			}
			if tc.Function.Name != "" {
				p.name = tc.Function.Name
			}
			p.arguments.WriteString(tc.Function.Arguments)
		}

		// Handle finish_reason
		if choice.FinishReason != nil {
			reason := *choice.FinishReason
			if reason == "tool_calls" && len(pendingCalls) > 0 {
				// Emit accumulated tool_calls
				if err := emitToolCallChunk(enc, model, pendingCalls, reason); err != nil {
					return err
				}
				pendingCalls = map[int]*pendingToolCall{}
				flushFn()
				continue
			}
			// Chunk ending with content text
			content := delta.Content
			resp := ChatResponse{
				Model:      model,
				CreatedAt:  time.Now(),
				Message:    OllamaMessage{Role: "assistant", Content: content},
				Done:       true,
				DoneReason: reason,
			}
			if err := enc.Encode(resp); err != nil {
				return fmt.Errorf("encoding final ollama chunk: %w", err)
			}
			flushFn()
			continue
		}

		// Regular chunk — emit content text (skip if only collecting tool_calls)
		if delta.Content != "" {
			resp := ChatResponse{
				Model:     model,
				CreatedAt: time.Now(),
				Message:   OllamaMessage{Role: "assistant", Content: delta.Content},
				Done:      false,
			}
			if err := enc.Encode(resp); err != nil {
				return fmt.Errorf("encoding ollama chunk: %w", err)
			}
			flushFn()
		}
	}

	return scanner.Err()
}

// emitToolCallChunk emits a ChatResponse with tool_calls assembled from pendingCalls.
func emitToolCallChunk(enc *json.Encoder, model string, pending map[int]*pendingToolCall, reason string) error {
	toolCalls := make([]ToolCall, 0, len(pending))
	for i := 0; i < len(pending); i++ {
		p, ok := pending[i]
		if !ok {
			continue
		}
		argStr := p.arguments.String()
		var args json.RawMessage
		if json.Valid([]byte(argStr)) {
			args = json.RawMessage(argStr)
		} else {
			args = json.RawMessage("{}")
		}
		toolCalls = append(toolCalls, ToolCall{
			Function: ToolCallFunction{
				Name:      p.name,
				Arguments: args,
			},
		})
	}
	resp := ChatResponse{
		Model:     model,
		CreatedAt: time.Now(),
		Message: OllamaMessage{
			Role:      "assistant",
			Content:   "",
			ToolCalls: toolCalls,
		},
		Done:       true,
		DoneReason: reason,
	}
	return enc.Encode(resp)
}

// ─── Generate: Ollama → OpenAI ────────────────────────────────────────────────

// ollamaGenerateToOpenAI converts an Ollama POST /api/generate request into an OpenAI
// POST /v1/completions request body.
func ollamaGenerateToOpenAI(req *GenerateRequest) *openAICompletionRequest {
	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	// Prepend the system prompt (if any) to the beginning of the prompt
	prompt := req.Prompt
	if req.System != "" && !req.Raw {
		prompt = req.System + "\n\n" + prompt
	}

	out := &openAICompletionRequest{
		Model:  req.Model,
		Prompt: prompt,
		Stream: stream,
	}
	if req.Suffix != "" {
		out.Suffix = req.Suffix
	}
	if o := req.Options; o != nil {
		out.Temperature = o.Temperature
		out.TopP = o.TopP
		out.MaxTokens = o.NumPredict
		out.Seed = o.Seed
	}
	out.ResponseFormat = ollamaFormatToOpenAI(req.Format)
	return out
}

// ─── Generate: OpenAI → Ollama (non-streaming) ────────────────────────────────

// openAIToOllamaGenerate converts a non-streaming /v1/completions response into
// an Ollama GenerateResponse.
func openAIToOllamaGenerate(model string, oai *openAICompletionResponse) *GenerateResponse {
	text := ""
	finishReason := "stop"
	if len(oai.Choices) > 0 {
		text = oai.Choices[0].Text
		if oai.Choices[0].FinishReason != "" {
			finishReason = oai.Choices[0].FinishReason
		}
	}
	return &GenerateResponse{
		Model:           model,
		CreatedAt:       time.Now(),
		Response:        text,
		Done:            true,
		DoneReason:      finishReason,
		PromptEvalCount: oai.Usage.PromptTokens,
		EvalCount:       oai.Usage.CompletionTokens,
	}
}

// ─── Generate: OpenAI → Ollama (streaming) ────────────────────────────────────

// streamOpenAIGenerateToOllama reads a /v1/completions SSE stream from src, converts
// each chunk to an Ollama GenerateResponse NDJSON format, and writes to dst.
func streamOpenAIGenerateToOllama(dst io.Writer, src io.Reader, model string) error {
	scanner := bufio.NewScanner(src)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	enc := json.NewEncoder(dst)

	flushFn := func() {
		if f, ok := dst.(interface{ Flush() }); ok {
			f.Flush()
		}
	}

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")

		if payload == "[DONE]" {
			final := GenerateResponse{
				Model:      model,
				CreatedAt:  time.Now(),
				Response:   "",
				Done:       true,
				DoneReason: "stop",
			}
			_ = enc.Encode(final)
			flushFn()
			return nil
		}

		var chunk openAICompletionStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		resp := GenerateResponse{
			Model:     model,
			CreatedAt: time.Now(),
			Response:  choice.Text,
			Done:      false,
		}

		if choice.FinishReason != nil {
			resp.Done = true
			resp.DoneReason = *choice.FinishReason
		}

		if err := enc.Encode(resp); err != nil {
			return fmt.Errorf("encoding generate chunk: %w", err)
		}
		flushFn()

		if resp.Done {
			return nil
		}
	}

	return scanner.Err()
}

// ─── Embed: Ollama → OpenAI ───────────────────────────────────────────────────

// ollamaEmbedToOpenAI converts an Ollama POST /api/embed request into an OpenAI
// POST /v1/embeddings request body.
func ollamaEmbedToOpenAI(req *EmbedRequest) *openAIEmbedRequest {
	return &openAIEmbedRequest{
		Model: req.Model,
		Input: req.Input,
	}
}

// ─── Embed: OpenAI → Ollama ───────────────────────────────────────────────────

// openAIToOllamaEmbed converts an OpenAI /v1/embeddings response into an Ollama EmbedResponse.
func openAIToOllamaEmbed(model string, oai *openAIEmbedResponse) *EmbedResponse {
	embeddings := make([][]float64, len(oai.Data))
	for i, d := range oai.Data {
		embeddings[i] = d.Embedding
	}
	return &EmbedResponse{
		Model:           model,
		Embeddings:      embeddings,
		PromptEvalCount: oai.Usage.PromptTokens,
	}
}
