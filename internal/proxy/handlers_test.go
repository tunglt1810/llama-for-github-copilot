package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/labstack/echo/v5"
)

func TestShowReturnsContextLength(t *testing.T) {
	e := echo.New()
	body := bytes.NewReader([]byte(`{"model":"test","verbose":true}`))
	req := httptest.NewRequest(http.MethodPost, "/api/show", body)
	req.Header.Set(echo.HeaderContentType, echo.MIMEApplicationJSON)
	rec := httptest.NewRecorder()
	c := e.NewContext(req, rec)

	h := &handlers{modelName: "llava", contextSize: 8192}
	if err := h.show(c); err != nil {
		t.Fatalf("show returned error: %v", err)
	}

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var resp ShowResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if !strings.Contains(resp.Modelfile, "PARAMETER num_ctx 8192") {
		t.Fatalf("expected modelfile to include context length, got %q", resp.Modelfile)
	}

	if strings.Contains(resp.Parameters, "num_ctx") {
		t.Fatalf("expected parameters to omit num_ctx, got %q", resp.Parameters)
	}

	value, ok := resp.ModelInfo["llama.context_length"]
	if !ok {
		t.Fatal("expected model_info to contain llama.context_length")
	}

	if got, ok := value.(float64); !ok || got != 8192 {
		t.Fatalf("expected llama.context_length 8192, got %v", value)
	}

	if value, ok := resp.ModelInfo["general.context_length"]; !ok {
		t.Fatal("expected model_info to contain general.context_length")
	} else if got, ok := value.(float64); !ok || got != 8192 {
		t.Fatalf("expected general.context_length 8192, got %v", value)
	}
}
