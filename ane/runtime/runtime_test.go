package runtime

import "testing"

func TestFirstByMap(t *testing.T) {
	tests := []struct {
		name       string
		available  map[string]bool
		candidates []string
		want       string
	}{
		{
			name:       "first match",
			available:  map[string]bool{"B": true, "A": true},
			candidates: []string{"A", "B", "C"},
			want:       "A",
		},
		{
			name:       "fallback match",
			available:  map[string]bool{"B": true},
			candidates: []string{"A", "B"},
			want:       "B",
		},
		{
			name:       "no match",
			available:  map[string]bool{"X": true},
			candidates: []string{"A", "B"},
			want:       "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := firstByMap(tt.available, tt.candidates)
			if got != tt.want {
				t.Fatalf("firstByMap() = %q, want %q", got, tt.want)
			}
		})
	}
}

func firstByMap(available map[string]bool, candidates []string) string {
	for _, c := range candidates {
		if available[c] {
			return c
		}
	}
	return ""
}
