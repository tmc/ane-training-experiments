// Package storiestrainer provides a Go ANE Stories training API.
//
// On darwin it prefers the bridge-backed trainer for full parity and falls back
// to direct-Go daemon-backed _ANEClient orchestration when bridge symbols are
// unavailable.
package storiestrainer
