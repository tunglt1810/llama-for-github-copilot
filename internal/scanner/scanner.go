// Package scanner discovers local GGUF model files from two sources:
//  1. The project-local models/ directory
//  2. The HuggingFace hub cache (~/.cache/huggingface/hub)
package scanner

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Model represents a discovered GGUF model file.
type Model struct {
	Path string
	Size int64
}

// Scan returns all available GGUF model files, sorted by path.
// It mirrors the scan_models() logic from the original start-server.sh.
func Scan(projectRoot string) ([]Model, error) {
	var models []Model

	// Source 1: project local models/ directory (maxdepth 1)
	localDir := filepath.Join(projectRoot, "models")
	entries, err := os.ReadDir(localDir)
	if err == nil {
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			if !strings.HasSuffix(e.Name(), ".gguf") {
				continue
			}
			full := filepath.Join(localDir, e.Name())
			info, err := os.Stat(full) // follows symlinks
			if err != nil || info.IsDir() {
				continue
			}
			models = append(models, Model{Path: full, Size: info.Size()})
		}
	}

	// Source 2: HuggingFace hub cache
	hfCacheDir := filepath.Join(os.Getenv("HOME"), ".cache", "huggingface", "hub")
	hfModels, err := scanHFCache(hfCacheDir)
	if err == nil {
		models = append(models, hfModels...)
	}

	// Deduplicate by real path (resolves symlinks)
	models = dedup(models)

	sort.Slice(models, func(i, j int) bool {
		return models[i].Path < models[j].Path
	})
	return models, nil
}

// scanHFCache scans models--*/ dirs in the HuggingFace cache,
// resolves refs/main → snapshots/<commit>/ and finds *.gguf files.
// Vision projector files (mmproj-* and *-projector*) are excluded.
func scanHFCache(cacheDir string) ([]Model, error) {
	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		return nil, err
	}

	var models []Model
	for _, e := range entries {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "models--") {
			continue
		}
		repoDir := filepath.Join(cacheDir, e.Name())
		refsFile := filepath.Join(repoDir, "refs", "main")

		data, err := os.ReadFile(refsFile)
		if err != nil {
			continue
		}
		commitHash := strings.TrimSpace(string(data))
		if commitHash == "" {
			continue
		}

		snapshotDir := filepath.Join(repoDir, "snapshots", commitHash)
		if _, err := os.Stat(snapshotDir); err != nil {
			continue
		}

		err = filepath.WalkDir(snapshotDir, func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			name := d.Name()
			if !strings.HasSuffix(name, ".gguf") {
				return nil
			}
			// Skip vision projector files
			if strings.HasPrefix(name, "mmproj-") || strings.Contains(name, "-projector") {
				return nil
			}
			info, err := os.Stat(path) // follows symlinks to confirm blob exists
			if err != nil || info.IsDir() {
				return nil
			}
			models = append(models, Model{Path: path, Size: info.Size()})
			return nil
		})
		if err != nil {
			continue
		}
	}
	return models, nil
}

// dedup removes duplicate model entries by resolving symlinks.
func dedup(models []Model) []Model {
	seen := make(map[string]struct{}, len(models))
	result := make([]Model, 0, len(models))
	for _, m := range models {
		real, err := filepath.EvalSymlinks(m.Path)
		if err != nil {
			real = m.Path
		}
		if _, ok := seen[real]; ok {
			continue
		}
		seen[real] = struct{}{}
		result = append(result, m)
	}
	return result
}
