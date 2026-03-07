package forward

import "testing"

func TestChannelFirstRoundTrip(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6}
	cf, err := ToChannelFirst(x, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	got, err := FromChannelFirst(cf, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	for i := range x {
		if x[i] != got[i] {
			t.Fatalf("round-trip mismatch at %d: got %v want %v", i, got[i], x[i])
		}
	}
}
