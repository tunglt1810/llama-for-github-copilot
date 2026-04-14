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

// ─── Helpers dùng chung ───────────────────────────────────────────────────────

// marshalBody serialise v thành JSON và bọc trong ReadCloser phù hợp với http.Request.Body.
func marshalBody(v any) (io.ReadCloser, int64, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return nil, 0, err
	}
	return io.NopCloser(bytes.NewReader(data)), int64(len(data)), nil
}

// toOpenAIContent chuyển đổi text và images thành json.RawMessage phù hợp với OpenAI format.
// Nếu không có images: trả về JSON string.
// Nếu có images: trả về JSON array of content parts (text + image_url).
func toOpenAIContent(text string, images []string) json.RawMessage {
	if len(images) == 0 {
		b, _ := json.Marshal(text)
		return json.RawMessage(b)
	}
	// Dùng anonymous struct để tránh import thêm
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
		// Nếu chưa có data URL prefix thì thêm vào
		if !strings.HasPrefix(img, "data:") {
			url = "data:image/png;base64," + img
		}
		parts = append(parts, imagePart{Type: "image_url", ImageURL: imageURL{URL: url}})
	}
	b, _ := json.Marshal(parts)
	return json.RawMessage(b)
}

// extractTextContent lấy nội dung text từ OpenAI content field (json.RawMessage).
// Hỗ trợ cả dạng string và dạng array of content parts.
func extractTextContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	// Thử string trước
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	// Thử array of parts — lấy tất cả text parts
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

// ollamaFormatToOpenAI chuyển đổi Ollama format field sang OpenAI response_format.
// "json" → {"type":"json_object"}
// JSON schema object → {"type":"json_schema","json_schema":<schema>}
func ollamaFormatToOpenAI(format json.RawMessage) *openAIResponseFormat {
	if len(format) == 0 {
		return nil
	}
	// Kiểm tra xem có phải là string "json" không
	var s string
	if err := json.Unmarshal(format, &s); err == nil {
		if s == "json" {
			return &openAIResponseFormat{Type: "json_object"}
		}
		return nil
	}
	// Là JSON object (JSON schema) → dùng json_schema type
	return &openAIResponseFormat{Type: "json_schema", Schema: format}
}

// ─── Chat: Ollama → OpenAI ────────────────────────────────────────────────────

// ollamaToOpenAI chuyển đổi Ollama POST /api/chat request thành OpenAI
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

	// Chuyển đổi messages — hỗ trợ images, tool_calls, tool role
	out.Messages = make([]openAIMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		msg := openAIMessage{
			Role:    m.Role,
			Content: toOpenAIContent(m.Content, m.Images),
		}
		// Chuyển Ollama tool_calls (arguments là json.RawMessage object) sang
		// OpenAI tool_calls (arguments là JSON-encoded string)
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
			// Khi có tool_calls, content thường là null/empty
			msg.Content = toOpenAIContent("", nil)
		}
		// Xử lý role "tool" — map tool_name sang name field
		if m.Role == "tool" && m.ToolName != "" {
			msg.Name = m.ToolName
			// Dùng tool_name làm fake tool_call_id vì Ollama không có field này
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

	// Chuyển tools
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

	// Chuyển format → response_format
	out.ResponseFormat = ollamaFormatToOpenAI(req.Format)

	return out
}

// ─── Chat: OpenAI → Ollama (non-streaming) ────────────────────────────────────

// openAIToOllamaResponse chuyển đổi non-streaming OpenAI response thành Ollama ChatResponse.
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
		// Chuyển OpenAI tool_calls (arguments là string) sang Ollama (arguments là json.RawMessage)
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

// pendingToolCall dùng để accumulate streaming tool_call chunks theo index.
type pendingToolCall struct {
	id        string
	name      string
	arguments strings.Builder
}

// streamOpenAIToOllama đọc OpenAI SSE stream từ src, chuyển đổi từng chunk sang
// Ollama NDJSON format và ghi vào dst.
//
// Hỗ trợ: text content, tool_calls (accumulate partial chunks).
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
	// Tăng buffer size để xử lý các chunk lớn
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	enc := json.NewEncoder(dst)

	// Map để accumulate tool_calls theo index
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
			// Nếu còn pending tool_calls chưa emit (do không có finish_reason chunk)
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

		// Accumulate tool_calls nếu có
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

		// Xử lý finish_reason
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
			// Chunk kết thúc với content text
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

		// Chunk thường — emit content text (bỏ qua nếu chỉ đang thu thập tool_calls)
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

// emitToolCallChunk emit một ChatResponse với tool_calls đã được assembl từ pendingCalls.
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

// ollamaGenerateToOpenAI chuyển đổi Ollama POST /api/generate request thành OpenAI
// POST /v1/completions request body.
func ollamaGenerateToOpenAI(req *GenerateRequest) *openAICompletionRequest {
	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	// Kết hợp system prompt (nếu có) vào đầu prompt
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

// openAIToOllamaGenerate chuyển đổi non-streaming /v1/completions response thành
// Ollama GenerateResponse.
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

// streamOpenAIGenerateToOllama đọc /v1/completions SSE stream từ src, chuyển đổi
// từng chunk sang Ollama GenerateResponse NDJSON format và ghi vào dst.
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

// ollamaEmbedToOpenAI chuyển đổi Ollama POST /api/embed request thành OpenAI
// POST /v1/embeddings request body.
func ollamaEmbedToOpenAI(req *EmbedRequest) *openAIEmbedRequest {
	return &openAIEmbedRequest{
		Model: req.Model,
		Input: req.Input,
	}
}

// ─── Embed: OpenAI → Ollama ───────────────────────────────────────────────────

// openAIToOllamaEmbed chuyển đổi OpenAI /v1/embeddings response thành Ollama EmbedResponse.
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
