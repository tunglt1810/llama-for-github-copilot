package flags

// Flag represents the name of a CLI parameter for llama-server.
// Use typed string constants to avoid hardcoding and typos.
type Flag string

const (
    FlagModel      Flag = "model"
    FlagAlias      Flag = "alias"
    FlagNgpuLayers Flag = "n-gpu-layers"
    FlagCtxSize    Flag = "ctx-size"
    FlagCtxPerSlot Flag = "ctx-per-slot"
    FlagParallel   Flag = "parallel"
    FlagCacheTypeK Flag = "cache-type-k"
    FlagCacheTypeV Flag = "cache-type-v"
    FlagFlashAttn  Flag = "flash-attn"
    FlagEmbedding  Flag = "embedding"
    FlagHost       Flag = "host"
    FlagPort       Flag = "port"
)

// Key returns the actual string used as the key in map[string]string.
func Key(f Flag) string {
    return string(f)
}
